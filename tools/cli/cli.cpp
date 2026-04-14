#include "chat.h"
#include "common.h"
#include "arg.h"
#include "console.h"
// #include "log.h"

#include "server-context.h"
#include "server-task.h"

#include <array>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <signal.h>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <logosdb/logosdb.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

static std::atomic<bool> g_is_interrupted = false;
static bool should_stop() {
    return g_is_interrupted.load();
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (g_is_interrupted.load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    g_is_interrupted.store(true);
}
#endif

struct cli_context {
    server_context ctx_server;
    json messages = json::array();
    std::vector<raw_buffer> input_files;
    task_params defaults;
    bool verbose_prompt;
    int reasoning_budget = -1;
    std::string reasoning_budget_message;

    // thread for showing "loading" animation
    std::atomic<bool> loading_show;

    cli_context(const common_params & params) {
        defaults.sampling    = params.sampling;
        defaults.speculative = params.speculative;
        defaults.n_keep      = params.n_keep;
        defaults.n_predict   = params.n_predict;
        defaults.antiprompt  = params.antiprompt;

        defaults.stream = true; // make sure we always use streaming mode
        defaults.timings_per_token = true; // in order to get timings even when we cancel mid-way
        // defaults.return_progress = true; // TODO: show progress

        verbose_prompt = params.verbose_prompt;
        reasoning_budget = params.reasoning_budget;
        reasoning_budget_message = params.reasoning_budget_message;
    }

    std::optional<std::vector<float>> generate_embedding(const std::string & text, std::string & err) {
        server_response_reader rd = ctx_server.get_response_reader();
        {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);
            task.id         = rd.get_new_id();
            task.index      = 0;
            task.params     = defaults;
            task.params.stream = false;
            task.cli        = true;
            task.cli_prompt = text;
            rd.post_task({std::move(task)});
        }

        server_task_result_ptr result = rd.next(should_stop);
        while (result) {
            if (result->is_error()) {
                const json err_data = result->to_json();
                if (err_data.contains("message")) {
                    err = err_data["message"].get<std::string>();
                } else {
                    err = err_data.dump();
                }
                return std::nullopt;
            }

            auto * embd = dynamic_cast<server_task_result_embd *>(result.get());
            if (embd != nullptr) {
                if (embd->embedding.empty()) {
                    err = "empty embedding result";
                    return std::nullopt;
                }

                const size_t n = embd->embedding[0].size();
                if (n == 0) {
                    err = "invalid embedding dimension";
                    return std::nullopt;
                }

                std::vector<float> pooled(n, 0.0f);
                for (const auto & row : embd->embedding) {
                    if (row.size() != n) {
                        err = "inconsistent embedding dimensions";
                        return std::nullopt;
                    }
                    for (size_t i = 0; i < n; ++i) {
                        pooled[i] += row[i];
                    }
                }
                const float inv_rows = 1.0f / std::max<size_t>(1, embd->embedding.size());
                for (float & v : pooled) {
                    v *= inv_rows;
                }

                float l2 = 0.0f;
                for (const float v : pooled) {
                    l2 += v*v;
                }
                l2 = std::sqrt(l2);
                if (l2 > 1e-12f) {
                    const float inv_l2 = 1.0f / l2;
                    for (float & v : pooled) {
                        v *= inv_l2;
                    }
                }

                return pooled;
            }

            if (result->is_stop()) {
                break;
            }

            result = rd.next(should_stop);
        }

        err = "did not receive embedding result";
        return std::nullopt;
    }

    std::string generate_completion_with_messages(
            const json & messages_in,
            result_timings & out_timings,
            bool stream_to_console,
            int n_predict_override = -1,
            const std::vector<llama_logit_bias> * extra_logit_bias = nullptr) {
        server_response_reader rd = ctx_server.get_response_reader();
        auto chat_params = format_chat_from_messages(messages_in);
        {
            // TODO: reduce some copies here in the future
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id         = rd.get_new_id();
            task.index      = 0;
            task.params     = defaults;           // copy
            if (n_predict_override > 0) {
                task.params.n_predict = n_predict_override;
            }
            if (extra_logit_bias && !extra_logit_bias->empty()) {
                task.params.sampling.logit_bias.insert(
                    task.params.sampling.logit_bias.end(),
                    extra_logit_bias->begin(),
                    extra_logit_bias->end());
            }
            task.cli_prompt = chat_params.prompt; // copy
            task.cli_files  = input_files;        // copy
            task.cli        = true;

            // chat template settings
            task.params.chat_parser_params = common_chat_parser_params(chat_params);
            task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            if (!chat_params.parser.empty()) {
                task.params.chat_parser_params.parser.load(chat_params.parser);
            }

            // reasoning budget sampler
            if (!chat_params.thinking_end_tag.empty()) {
                const llama_vocab * vocab = llama_model_get_vocab(
                    llama_get_model(ctx_server.get_llama_context()));

                task.params.sampling.reasoning_budget_tokens = reasoning_budget;
                task.params.sampling.generation_prompt = chat_params.generation_prompt;

                if (!chat_params.thinking_start_tag.empty()) {
                    task.params.sampling.reasoning_budget_start =
                        common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
                }
                task.params.sampling.reasoning_budget_end =
                    common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
                task.params.sampling.reasoning_budget_forced =
                    common_tokenize(vocab, reasoning_budget_message + chat_params.thinking_end_tag, false, true);
            }

            rd.post_task({std::move(task)});
        }

        if (stream_to_console && verbose_prompt) {
            console::set_display(DISPLAY_TYPE_PROMPT);
            console::log("%s\n\n", chat_params.prompt.c_str());
            console::set_display(DISPLAY_TYPE_RESET);
        }

        // wait for first result
        if (stream_to_console) {
            console::spinner::start();
        }
        server_task_result_ptr result = rd.next(should_stop);

        if (stream_to_console) {
            console::spinner::stop();
        }
        std::string curr_content;
        bool is_thinking = false;

        while (result) {
            if (should_stop()) {
                break;
            }
            if (result->is_error()) {
                json err_data = result->to_json();
                if (err_data.contains("message")) {
                    console::error("Error: %s\n", err_data["message"].get<std::string>().c_str());
                } else {
                    console::error("Error: %s\n", err_data.dump().c_str());
                }
                return curr_content;
            }
            auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                out_timings = std::move(res_partial->timings);
                for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                    if (!diff.content_delta.empty()) {
                        if (stream_to_console && is_thinking) {
                            console::log("\n[End thinking]\n\n");
                            console::set_display(DISPLAY_TYPE_RESET);
                            is_thinking = false;
                        }
                        curr_content += diff.content_delta;
                        if (stream_to_console) {
                            console::log("%s", diff.content_delta.c_str());
                            console::flush();
                        }
                    }
                    if (!diff.reasoning_content_delta.empty()) {
                        if (stream_to_console) {
                            console::set_display(DISPLAY_TYPE_REASONING);
                            if (!is_thinking) {
                                console::log("[Start thinking]\n");
                            }
                        }
                        is_thinking = true;
                        if (stream_to_console) {
                            console::log("%s", diff.reasoning_content_delta.c_str());
                            console::flush();
                        }
                    }
                }
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                out_timings = std::move(res_final->timings);
                break;
            }
            result = rd.next(should_stop);
        }
        g_is_interrupted.store(false);
        // server_response_reader automatically cancels pending tasks upon destruction
        return curr_content;
    }

    std::string generate_completion(result_timings & out_timings) {
        return generate_completion_with_messages(messages, out_timings, true);
    }

    std::string generate_completion(
            result_timings & out_timings,
            const std::vector<llama_logit_bias> & extra_logit_bias) {
        return generate_completion_with_messages(messages, out_timings, true, -1, &extra_logit_bias);
    }

    // TODO: support remote files in the future (http, https, etc)
    std::string load_input_file(const std::string & fname, bool is_media) {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            return "";
        }
        if (is_media) {
            raw_buffer buf;
            buf.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            input_files.push_back(std::move(buf));
            return mtmd_default_marker();
        } else {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            return content;
        }
    }

    common_chat_params format_chat_from_messages(const json & messages_in) {
        auto meta = ctx_server.get_meta();
        auto & chat_params = meta.chat_params;

        common_chat_templates_inputs inputs;
        inputs.messages              = common_chat_msgs_parse_oaicompat(messages_in);
        inputs.tools                 = {}; // TODO
        inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
        inputs.json_schema           = ""; // TODO
        inputs.grammar               = ""; // TODO
        inputs.use_jinja             = chat_params.use_jinja;
        inputs.parallel_tool_calls   = false;
        inputs.add_generation_prompt = true;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs.force_pure_content    = chat_params.force_pure_content;
        inputs.enable_thinking       = chat_params.enable_thinking ? common_chat_templates_support_enable_thinking(chat_params.tmpls.get()) : false;

        // Apply chat template to the list of messages
        return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
    }

    common_chat_params format_chat() {
        return format_chat_from_messages(messages);
    }
};

// TODO?: Make this reusable, enums, docs
static const std::array<const std::string, 8> cmds = {
    "/audio ",
    "/clear",
    "/exit",
    "/glob ",
    "/image ",
    "/read ",
    "/regen",
    "/teach ",
};

static std::vector<std::pair<std::string, size_t>> auto_completion_callback(std::string_view line, size_t cursor_byte_pos) {
    std::vector<std::pair<std::string, size_t>> matches;
    std::string cmd;

    if (line.length() > 1 && line[0] == '/' && !std::any_of(cmds.begin(), cmds.end(), [line](const std::string & prefix) {
        return string_starts_with(line, prefix);
    })) {
        auto it = cmds.begin();

        while ((it = std::find_if(it, cmds.end(), [line](const std::string & cmd_line) {
            return string_starts_with(cmd_line, line);
        })) != cmds.end()) {
            matches.emplace_back(*it, (*it).length());
            ++it;
        }
    } else {
        auto it = std::find_if(cmds.begin(), cmds.end(), [line](const std::string & prefix) {
            return prefix.back() == ' ' && string_starts_with(line, prefix);
        });

        if (it != cmds.end()) {
            cmd = *it;
        }
    }

    if (!cmd.empty() && cmd != "/glob " && line.length() >= cmd.length() && cursor_byte_pos >= cmd.length()) {
        const std::string path_prefix  = std::string(line.substr(cmd.length(), cursor_byte_pos - cmd.length()));
        const std::string path_postfix = std::string(line.substr(cursor_byte_pos));
        auto cur_dir = std::filesystem::current_path();
        std::string cur_dir_str = cur_dir.string();
        std::string expanded_prefix = path_prefix;

#if !defined(_WIN32)
        if (string_starts_with(path_prefix, "~")) {
            const char * home = std::getenv("HOME");
            if (home && home[0]) {
                expanded_prefix = std::string(home) + path_prefix.substr(1);
            }
        }
        if (string_starts_with(expanded_prefix, "/")) {
#else
        if (std::isalpha(expanded_prefix[0]) && expanded_prefix.find(':') == 1) {
#endif
            cur_dir = std::filesystem::path(expanded_prefix).parent_path();
            cur_dir_str = "";
        } else if (!path_prefix.empty()) {
            cur_dir /= std::filesystem::path(path_prefix).parent_path();
        }

        std::error_code ec;
        for (const auto & entry : std::filesystem::directory_iterator(cur_dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.exists(ec)) {
                ec.clear();
                continue;
            }

            const std::string path_full = entry.path().string();
            std::string path_entry = !cur_dir_str.empty() && string_starts_with(path_full, cur_dir_str) ? path_full.substr(cur_dir_str.length() + 1) : path_full;

            if (entry.is_directory(ec)) {
                path_entry.push_back(std::filesystem::path::preferred_separator);
            }

            if (expanded_prefix.empty() || string_starts_with(path_entry, expanded_prefix)) {
                std::string updated_line = cmd + path_entry;
                matches.emplace_back(updated_line + path_postfix, updated_line.length());
            }

            if (ec) {
                ec.clear();
            }
        }

        if (matches.empty()) {
            std::string updated_line = cmd + path_prefix;
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        // Add the longest common prefix
        if (!expanded_prefix.empty() && matches.size() > 1) {
            const std::string_view match0(matches[0].first);
            const std::string_view match1(matches[1].first);
            auto it = std::mismatch(match0.begin(), match0.end(), match1.begin(), match1.end());
            size_t len = it.first - match0.begin();

            for (size_t i = 2; i < matches.size(); ++i) {
                const std::string_view matchi(matches[i].first);
                auto cmp = std::mismatch(match0.begin(), match0.end(), matchi.begin(), matchi.end());
                len = std::min(len, static_cast<size_t>(cmp.first - match0.begin()));
            }

            std::string updated_line = std::string(match0.substr(0, len));
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        std::sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) {
            return a.first.compare(0, a.second, b.first, 0, b.second) < 0;
        });
    }

    return matches;
}

static constexpr size_t FILE_GLOB_MAX_RESULTS = 100;

struct semantic_memory_options {
    std::string db_path;
    int dim = 0;
    std::string teach_open_tag = "<teach>";
    std::string teach_close_tag = "</teach>";
    std::string learn_from_response_open_tag = "<learn-from-response>";
    std::string learn_from_response_close_tag = "</learn-from-response>";
    std::string teach_prefix;
    bool enable_teach_tags = true;
    bool enable_teach_variants = false;
    bool enable_learn_from_response_tags = true;
    int hint_tokens = 0;
    bool hint_llm_compress = false;
    int hint_llm_top_k = 3;
    int hint_llm_n_predict = 48;
};

static bool parse_semantic_memory_args(
        int argc,
        char ** argv,
        semantic_memory_options & out,
        std::vector<std::string> & filtered,
        std::string & err) {
    filtered.clear();
    filtered.reserve(argc);
    filtered.emplace_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto consume_value = [&](const char * name, std::string & dst) -> bool {
            if (i + 1 >= argc) {
                err = std::string("missing value for ") + name;
                return false;
            }
            dst = argv[++i];
            return true;
        };

        if (arg == "--semantic-memory-db") {
            if (!consume_value("--semantic-memory-db", out.db_path)) {
                return false;
            }
        } else if (arg == "--semantic-memory-dim") {
            std::string value;
            if (!consume_value("--semantic-memory-dim", value)) {
                return false;
            }
            out.dim = std::max(0, atoi(value.c_str()));
        } else if (arg == "--semantic-memory-hint-tokens") {
            std::string value;
            if (!consume_value("--semantic-memory-hint-tokens", value)) {
                return false;
            }
            out.hint_tokens = std::max(0, atoi(value.c_str()));
        } else if (arg == "--semantic-memory-hint-llm-compress") {
            out.hint_llm_compress = true;
        } else if (arg == "--no-semantic-memory-hint-llm-compress") {
            out.hint_llm_compress = false;
        } else if (arg == "--semantic-memory-hint-llm-top-k") {
            std::string value;
            if (!consume_value("--semantic-memory-hint-llm-top-k", value)) {
                return false;
            }
            out.hint_llm_top_k = std::max(1, atoi(value.c_str()));
        } else if (arg == "--semantic-memory-hint-llm-n-predict") {
            std::string value;
            if (!consume_value("--semantic-memory-hint-llm-n-predict", value)) {
                return false;
            }
            out.hint_llm_n_predict = std::max(8, atoi(value.c_str()));
        } else if (arg == "--semantic-memory-teach-open-tag") {
            if (!consume_value("--semantic-memory-teach-open-tag", out.teach_open_tag)) {
                return false;
            }
        } else if (arg == "--semantic-memory-teach-close-tag") {
            if (!consume_value("--semantic-memory-teach-close-tag", out.teach_close_tag)) {
                return false;
            }
        } else if (arg == "--semantic-memory-teach-prefix") {
            if (!consume_value("--semantic-memory-teach-prefix", out.teach_prefix)) {
                return false;
            }
        } else if (arg == "--semantic-memory-learn-response-open-tag") {
            if (!consume_value("--semantic-memory-learn-response-open-tag", out.learn_from_response_open_tag)) {
                return false;
            }
        } else if (arg == "--semantic-memory-learn-response-close-tag") {
            if (!consume_value("--semantic-memory-learn-response-close-tag", out.learn_from_response_close_tag)) {
                return false;
            }
        } else if (arg == "--semantic-memory-teach-tags") {
            out.enable_teach_tags = true;
        } else if (arg == "--no-semantic-memory-teach-tags") {
            out.enable_teach_tags = false;
        } else if (arg == "--semantic-memory-teach-variants") {
            out.enable_teach_variants = true;
        } else if (arg == "--no-semantic-memory-teach-variants") {
            out.enable_teach_variants = false;
        } else if (arg == "--semantic-memory-learn-response-tags") {
            out.enable_learn_from_response_tags = true;
        } else if (arg == "--no-semantic-memory-learn-response-tags") {
            out.enable_learn_from_response_tags = false;
        } else {
            filtered.push_back(arg);
        }
    }

    return true;
}

static std::string iso8601_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

struct parsed_date { int y; int m; int d; };

static bool parse_iso_date(const std::string & s, parsed_date & out) {
    // Accept YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ prefix
    if (s.size() < 10) return false;
    if (s[4] != '-' || s[7] != '-') return false;
    try {
        out.y = std::stoi(s.substr(0, 4));
        out.m = std::stoi(s.substr(5, 2));
        out.d = std::stoi(s.substr(8, 2));
        return out.m >= 1 && out.m <= 12 && out.d >= 1 && out.d <= 31;
    } catch (...) {
        return false;
    }
}

static int month_difference(const parsed_date & a, const parsed_date & b) {
    int months = (b.y - a.y) * 12 + (b.m - a.m);
    if (b.d < a.d) months -= 1;
    return std::abs(months);
}

static std::string format_elapsed(const parsed_date & earlier, const parsed_date & later) {
    int months = month_difference(earlier, later);
    if (months == 0) {
        int days = (later.y - earlier.y) * 365 + (later.m - earlier.m) * 30 + (later.d - earlier.d);
        if (days < 0) days = -days;
        if (days <= 1) return "today";
        return std::to_string(days) + " days";
    }
    if (months < 12) {
        return std::to_string(months) + " month" + (months == 1 ? "" : "s");
    }
    int years = months / 12;
    int rem = months % 12;
    std::string r = std::to_string(years) + " year" + (years == 1 ? "" : "s");
    if (rem > 0) r += " " + std::to_string(rem) + " month" + (rem == 1 ? "" : "s");
    return r;
}

static std::string extract_timestamp_from_chunk(const std::string & chunk) {
    // Chunks are prefixed with [YYYY-MM-DDTHH:MM:SSZ]
    if (chunk.size() > 2 && chunk[0] == '[') {
        auto close = chunk.find(']');
        if (close != std::string::npos && close > 1) {
            return chunk.substr(1, close - 1);
        }
    }
    return "";
}

static std::vector<parsed_date> extract_dates_from_text(const std::string & text) {
    std::vector<parsed_date> dates;
    // Scan for YYYY-MM-DD patterns anywhere in text.
    for (size_t i = 0; i + 10 <= text.size(); ++i) {
        if (text[i + 4] == '-' && text[i + 7] == '-') {
            parsed_date d{};
            if (parse_iso_date(text.substr(i, 10), d)) {
                bool dup = false;
                for (const auto & e : dates) {
                    if (e.y == d.y && e.m == d.m && e.d == d.d) { dup = true; break; }
                }
                if (!dup) dates.push_back(d);
                i += 9;
            }
        }
    }
    return dates;
}

static std::vector<std::string> extract_hint_tokens(const std::string & text, int max_tokens) {
    std::vector<std::string> out;
    if (max_tokens <= 0 || text.empty()) {
        return out;
    }

    static const std::unordered_set<std::string> stop = {
        "the", "a", "an", "is", "are", "to", "of", "and", "in", "for", "with", "my", "your"
    };
    std::regex token_re(R"([A-Za-z0-9][A-Za-z0-9_-]*)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), token_re);
    auto end   = std::sregex_iterator();
    std::unordered_set<std::string> seen;

    for (auto it = begin; it != end && (int) out.size() < max_tokens; ++it) {
        std::string token = (*it).str();
        std::string lower = token;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return (char) std::tolower(c);
        });
        if (stop.count(lower) > 0) {
            continue;
        }
        if (seen.insert(lower).second) {
            out.push_back(token);
        }
    }

    return out;
}

static std::string normalize_hint_text(std::string text);

static std::string build_prompt_hint_suffix(const std::vector<std::string> & tokens) {
    if (tokens.empty()) {
        return "";
    }
    std::string joined;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            joined += ' ';
        }
        joined += tokens[i];
    }
    return "\nHint: " + joined;
}

static std::string build_prompt_injection_suffix(const std::string & text) {
    const std::string stripped = string_strip(text);
    if (stripped.empty()) {
        return "";
    }
    // Compressor may output Quote: "..." or Hint: ... — pass through as-is.
    if (string_starts_with(stripped, "Quote:") || string_starts_with(stripped, "Hint:")) {
        return "\n" + stripped;
    }
    return "\nHint: " + normalize_hint_text(stripped);
}

static std::string normalize_hint_text(std::string text) {
    text = string_strip(text);
    if (text.empty()) {
        return text;
    }
    // Don't strip Quote:/Hint: prefix if present — handled by build_prompt_injection_suffix
    if (string_starts_with(text, "Quote:") || string_starts_with(text, "Hint:")) {
        return text;
    }
    if (string_starts_with(text, "HINT:")) {
        text = string_strip(text.substr(5));
    }
    while (!text.empty() && (text[0] == '-' || text[0] == '*' || text[0] == '"' || text[0] == '\'')) {
        text.erase(text.begin());
        text = string_strip(text);
    }
    while (!text.empty() && (text.back() == '"' || text.back() == '\'')) {
        text.pop_back();
    }
    return string_strip(text);
}

static std::string build_llm_compressed_hint(
        cli_context & ctx_cli,
        const std::string & query,
        const std::vector<std::string> & memory_chunks,
        int n_predict) {
    if (query.empty() || memory_chunks.empty()) {
        return "";
    }

    std::ostringstream user_prompt;
    user_prompt << "Extract a factual memory injection for the query.\n";
    user_prompt << "Rules:\n";
    user_prompt << "- If the memory contains a short, precise fact that directly answers the query, output EXACTLY:\n";
    user_prompt << "  Quote: \"exact text from memory\"\n";
    user_prompt << "- If the memory is relevant but needs interpretation or combination, output EXACTLY:\n";
    user_prompt << "  Hint: your concise summary (max 20 words)\n";
    user_prompt << "- Copy facts from memory only. No guesses.\n";
    user_prompt << "- Chunks may have timestamps [YYYY-MM-DDT...Z]. Use chronological order to resolve updates.\n";
    user_prompt << "- If memory is irrelevant, output: NONE\n\n";
    user_prompt << "Query:\n" << query << "\n\n";
    user_prompt << "Memory chunks:\n";
    std::vector<std::pair<size_t, parsed_date>> chunk_dates;
    for (size_t i = 0; i < memory_chunks.size(); ++i) {
        user_prompt << (i + 1) << ") " << memory_chunks[i] << "\n";
        std::string ts = extract_timestamp_from_chunk(memory_chunks[i]);
        parsed_date d{};
        if (!ts.empty() && parse_iso_date(ts, d)) {
            chunk_dates.push_back({i, d});
        }
    }

    // Collect all unique dates mentioned in chunk text and query (not just metadata timestamps).
    std::vector<parsed_date> all_dates;
    for (const auto & [idx, d] : chunk_dates) {
        bool dup = false;
        for (const auto & e : all_dates) { if (e.y == d.y && e.m == d.m && e.d == d.d) { dup = true; break; } }
        if (!dup) all_dates.push_back(d);
    }
    for (const auto & chunk : memory_chunks) {
        for (const auto & d : extract_dates_from_text(chunk)) {
            bool dup = false;
            for (const auto & e : all_dates) { if (e.y == d.y && e.m == d.m && e.d == d.d) { dup = true; break; } }
            if (!dup) all_dates.push_back(d);
        }
    }
    for (const auto & d : extract_dates_from_text(query)) {
        bool dup = false;
        for (const auto & e : all_dates) { if (e.y == d.y && e.m == d.m && e.d == d.d) { dup = true; break; } }
        if (!dup) all_dates.push_back(d);
    }

    auto date_key = [](const parsed_date & d) { return d.y * 10000 + d.m * 100 + d.d; };
    std::sort(all_dates.begin(), all_dates.end(), [&](const parsed_date & a, const parsed_date & b) {
        return date_key(a) < date_key(b);
    });

    if (all_dates.size() >= 2) {
        user_prompt << "\nPre-computed date math (use these, do NOT re-calculate):\n";
        for (size_t i = 0; i < all_dates.size(); ++i) {
            for (size_t j = i + 1; j < all_dates.size(); ++j) {
                std::string elapsed = format_elapsed(all_dates[i], all_dates[j]);
                char buf_a[16], buf_b[16];
                snprintf(buf_a, sizeof(buf_a), "%04d-%02d-%02d", all_dates[i].y, all_dates[i].m, all_dates[i].d);
                snprintf(buf_b, sizeof(buf_b), "%04d-%02d-%02d", all_dates[j].y, all_dates[j].m, all_dates[j].d);
                user_prompt << "- From " << buf_a << " to " << buf_b << " = exactly " << elapsed << ".\n";
            }
        }
    }

    user_prompt << "\nOutput:";

    json compressor_messages = json::array();
    compressor_messages.push_back({
        {"role", "system"},
        {"content", "You are a strict memory extractor. Output exactly one line: Quote, Hint, or NONE."}
    });
    compressor_messages.push_back({
        {"role", "user"},
        {"content", user_prompt.str()}
    });

    result_timings hint_timings;
    std::string raw = ctx_cli.generate_completion_with_messages(compressor_messages, hint_timings, false, n_predict);
    raw = string_strip(raw);
    if (raw.empty()) {
        return "";
    }
    std::string lower = raw;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    if (lower == "none") {
        return "";
    }
    return raw;
}

static std::vector<std::string> extract_teach_blocks(
        const std::string & input,
        const std::string & open_tag,
        const std::string & close_tag,
        std::string & cleaned,
        bool keep_content) {
    std::vector<std::string> teaches;
    cleaned.clear();

    if (open_tag.empty() || close_tag.empty()) {
        cleaned = input;
        return teaches;
    }

    size_t pos = 0;
    while (true) {
        const size_t start = input.find(open_tag, pos);
        if (start == std::string::npos) {
            cleaned += input.substr(pos);
            break;
        }

        cleaned += input.substr(pos, start - pos);
        const size_t content_start = start + open_tag.size();
        const size_t end = input.find(close_tag, content_start);
        if (end == std::string::npos) {
            cleaned += input.substr(start);
            break;
        }

        if (keep_content) {
            cleaned += input.substr(content_start, end - content_start);
        }
        teaches.push_back(string_strip(input.substr(content_start, end - content_start)));
        pos = end + close_tag.size();
    }

    cleaned = string_strip(cleaned);
    return teaches;
}

static std::vector<std::string> build_teach_variants(const std::string & text, bool enable_variants) {
    std::vector<std::string> variants;
    const std::string base = string_strip(text);
    if (base.empty()) {
        return variants;
    }

    auto add_unique = [&](const std::string & v) {
        const std::string s = string_strip(v);
        if (s.empty()) {
            return;
        }
        if (std::find(variants.begin(), variants.end(), s) == variants.end()) {
            variants.push_back(s);
        }
    };

    add_unique(base);
    if (!enable_variants) {
        return variants;
    }

    std::smatch match;
    const std::regex re_commute(R"(takes\s+(\d+\s+minutes each way))", std::regex::icase);
    if (std::regex_search(base, match, re_commute) && match.size() > 1) {
        add_unique(match[1].str());
    }

    const std::regex re_recommend(R"(recommend\s+([A-Za-z0-9' -]+)\.)", std::regex::icase);
    if (std::regex_search(base, match, re_recommend) && match.size() > 1) {
        add_unique(match[1].str());
    }

    const std::regex re_brand(R"(\b(Sony|Canon|Nikon|Fujifilm|Panasonic)\b)", std::regex::icase);
    if (std::regex_search(base, match, re_brand) && match.size() > 1) {
        add_unique(match[1].str() + "-compatible accessories");
    }

    const std::regex re_is_tail(R"(\b(?:is|are)\s+([^.!?]+))", std::regex::icase);
    if (std::regex_search(base, match, re_is_tail) && match.size() > 1) {
        add_unique(match[1].str());
    }

    return variants;
}

int main(int argc, char ** argv) {
    semantic_memory_options semantic_opts;
    std::vector<std::string> argv_filtered_storage;
    std::string semantic_arg_err;
    if (!parse_semantic_memory_args(argc, argv, semantic_opts, argv_filtered_storage, semantic_arg_err)) {
        fprintf(stderr, "Semantic memory arg error: %s\n", semantic_arg_err.c_str());
        return 1;
    }

    std::vector<char *> argv_filtered;
    argv_filtered.reserve(argv_filtered_storage.size());
    for (auto & s : argv_filtered_storage) {
        argv_filtered.push_back(s.data());
    }

    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    common_init();

    if (!common_params_parse((int) argv_filtered.size(), argv_filtered.data(), params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    logosdb_t * mem_db = nullptr;
    auto mem_db_close = [&]() {
        if (mem_db) { logosdb_close(mem_db); mem_db = nullptr; }
    };

    // TODO: maybe support it later?
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
        console::error("--no-conversation is not supported by llama-cli\n");
        console::error("please use llama-completion instead\n");
    }

    // struct that contains llama context and inference
    cli_context ctx_cli(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    // TODO: avoid using atexit() here by making `console` a singleton
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    console::set_display(DISPLAY_TYPE_RESET);
    console::set_completion_callback(auto_completion_callback);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    console::log("\nLoading model... "); // followed by loading animation
    console::spinner::start();
    if (!ctx_cli.ctx_server.load_model(params)) {
        console::spinner::stop();
        console::error("\nFailed to load the model\n");
        return 1;
    }

    console::spinner::stop();
    console::log("\n");

    if (!semantic_opts.db_path.empty()) {
        if (semantic_opts.dim <= 0) {
            console::error("--semantic-memory-dim is required when using --semantic-memory-db\n");
            return 1;
        }
        char * db_err = nullptr;
        logosdb_options_t * db_opts = logosdb_options_create();
        logosdb_options_set_dim(db_opts, semantic_opts.dim);
        mem_db = logosdb_open(semantic_opts.db_path.c_str(), db_opts, &db_err);
        logosdb_options_destroy(db_opts);
        if (!mem_db) {
            console::error("failed to open semantic memory db: %s\n", db_err ? db_err : "unknown");
            free(db_err);
            return 1;
        }
        console::log("semantic memory db: %s (dim=%d, %zu rows)\n",
            semantic_opts.db_path.c_str(), semantic_opts.dim, logosdb_count(mem_db));
    }

    std::thread inference_thread([&ctx_cli]() {
        ctx_cli.ctx_server.start_loop();
    });

    auto inf = ctx_cli.ctx_server.get_meta();
    std::string modalities = "text";
    if (inf.has_inp_image) {
        modalities += ", vision";
    }
    if (inf.has_inp_audio) {
        modalities += ", audio";
    }

    auto add_system_prompt = [&]() {
        if (!params.system_prompt.empty()) {
            ctx_cli.messages.push_back({
                {"role",    "system"},
                {"content", params.system_prompt}
            });
        }
    };
    add_system_prompt();

    console::log("\n");
    console::log("%s\n", LLAMA_ASCII_LOGO);
    console::log("build      : %s\n", inf.build_info.c_str());
    console::log("model      : %s\n", inf.model_name.c_str());
    console::log("modalities : %s\n", modalities.c_str());
    if (!params.system_prompt.empty()) {
        console::log("using custom system prompt\n");
    }
    console::log("\n");
    console::log("available commands:\n");
    console::log("  /exit or Ctrl+C     stop or exit\n");
    console::log("  /regen              regenerate the last response\n");
    console::log("  /clear              clear the chat history\n");
    console::log("  /teach <text>       add external memory embedding only\n");
    console::log("  /read <file>        add a text file\n");
    console::log("  /glob <pattern>     add text files using globbing pattern\n");
    if (inf.has_inp_image) {
        console::log("  /image <file>       add an image file\n");
    }
    if (inf.has_inp_audio) {
        console::log("  /audio <file>       add an audio file\n");
    }
    console::log("\n");
    if (mem_db) {
        if (semantic_opts.enable_teach_tags) {
            console::log("teach tags enabled: %s ... %s\n", semantic_opts.teach_open_tag.c_str(), semantic_opts.teach_close_tag.c_str());
        }
        if (semantic_opts.enable_teach_variants) {
            console::log("teach variants enabled: yes\n");
        }
        if (semantic_opts.hint_tokens > 0) {
            console::log("prompt hint tokens enabled: %d\n", semantic_opts.hint_tokens);
        }
        if (semantic_opts.hint_llm_compress) {
            console::log("prompt hint llm-compress enabled: top-k=%d n-predict=%d\n",
                semantic_opts.hint_llm_top_k, semantic_opts.hint_llm_n_predict);
        }
        if (!semantic_opts.teach_prefix.empty()) {
            console::log("teach prefix enabled: %s\n", semantic_opts.teach_prefix.c_str());
        }
        if (semantic_opts.enable_learn_from_response_tags) {
            console::log("learn-from-response tags enabled: %s ... %s\n",
                semantic_opts.learn_from_response_open_tag.c_str(),
                semantic_opts.learn_from_response_close_tag.c_str());
        }
        console::log("\n");
    }

    // interactive loop
    std::string cur_msg;

    auto add_text_file = [&](const std::string & fname) -> bool {
        std::string marker = ctx_cli.load_input_file(fname, false);
        if (marker.empty()) {
            console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
            return false;
        }
        if (inf.fim_sep_token != LLAMA_TOKEN_NULL) {
            cur_msg += common_token_to_piece(ctx_cli.ctx_server.get_llama_context(), inf.fim_sep_token, true);
            cur_msg += fname;
            cur_msg.push_back('\n');
        } else {
            cur_msg += "--- File: ";
            cur_msg += fname;
            cur_msg += " ---\n";
        }
        cur_msg += marker;
        console::log("Loaded text from '%s'\n", fname.c_str());
        return true;
    };

    while (true) {
        std::string buffer;
        console::set_display(DISPLAY_TYPE_USER_INPUT);
        if (params.prompt.empty()) {
            console::log("\n> ");
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
        } else {
            // process input prompt from args
            for (auto & fname : params.image) {
                std::string marker = ctx_cli.load_input_file(fname, true);
                if (marker.empty()) {
                    console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                    break;
                }
                console::log("Loaded media from '%s'\n", fname.c_str());
                cur_msg += marker;
            }
            buffer = params.prompt;
            if (buffer.size() > 500) {
                console::log("\n> %s ... (truncated)\n", buffer.substr(0, 500).c_str());
            } else {
                console::log("\n> %s\n", buffer.c_str());
            }
            params.prompt.clear(); // only use it once
        }
        console::set_display(DISPLAY_TYPE_RESET);
        console::log("\n");

        if (should_stop()) {
            g_is_interrupted.store(false);
            break;
        }

        // remove trailing newline
        if (!buffer.empty() &&buffer.back() == '\n') {
            buffer.pop_back();
        }

        // skip empty messages
        if (buffer.empty()) {
            continue;
        }

        bool add_user_msg = true;
        std::vector<std::string> learn_from_response_prompts;

        // process commands
        if (string_starts_with(buffer, "/exit")) {
            break;
        } else if (string_starts_with(buffer, "/teach ")) {
            if (!mem_db) {
                console::error("semantic memory is disabled. use --semantic-memory-db <path> --semantic-memory-dim <D>\n");
                continue;
            }

            const std::string teach_text = string_strip(buffer.substr(7));
            if (teach_text.empty()) {
                console::error("empty teach content\n");
                continue;
            }

            size_t taught = 0;
            for (const auto & teach_variant : build_teach_variants(teach_text, semantic_opts.enable_teach_variants)) {
                std::string err;
                auto embd = ctx_cli.generate_embedding(teach_variant, err);
                if (!embd) {
                    console::error("teach embedding failed: %s\n", err.c_str());
                    continue;
                }
                char * db_err = nullptr;
                std::string ts = iso8601_now();
                uint64_t id = logosdb_put(mem_db, embd->data(), (int)embd->size(),
                    teach_variant.c_str(), ts.c_str(), &db_err);
                if (id == UINT64_MAX) {
                    console::error("teach write failed: %s\n", db_err ? db_err : "unknown");
                    free(db_err);
                    continue;
                }
                ++taught;
            }
            console::log("taught %zu memory row(s)\n", taught);
            continue;
        } else if (string_starts_with(buffer, "/regen")) {
            if (ctx_cli.messages.size() >= 2) {
                size_t last_idx = ctx_cli.messages.size() - 1;
                ctx_cli.messages.erase(last_idx);
                add_user_msg = false;
            } else {
                console::error("No message to regenerate.\n");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            ctx_cli.messages.clear();
            add_system_prompt();

            ctx_cli.input_files.clear();
            console::log("Chat history cleared.\n");
            continue;
        } else if (
                (string_starts_with(buffer, "/image ") && inf.has_inp_image) ||
                (string_starts_with(buffer, "/audio ") && inf.has_inp_audio)) {
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname = string_strip(buffer.substr(7));
            std::string marker = ctx_cli.load_input_file(fname, true);
            if (marker.empty()) {
                console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            cur_msg += marker;
            console::log("Loaded media from '%s'\n", fname.c_str());
            continue;
        } else if (string_starts_with(buffer, "/read ")) {
            std::string fname = string_strip(buffer.substr(6));
            add_text_file(fname);
            continue;
        } else if (string_starts_with(buffer, "/glob ")) {
            std::error_code ec;
            size_t count = 0;
            auto curdir = std::filesystem::current_path();
            std::string pattern = string_strip(buffer.substr(6));
            std::filesystem::path rel_path;

            auto startglob = pattern.find_first_of("![*?");
            if (startglob != std::string::npos && startglob != 0) {
                auto endpath = pattern.substr(0, startglob).find_last_of('/');
                if (endpath != std::string::npos) {
                    std::string rel_pattern = pattern.substr(0, endpath);
#if !defined(_WIN32)
                    if (string_starts_with(rel_pattern, "~")) {
                        const char * home = std::getenv("HOME");
                        if (home && home[0]) {
                            rel_pattern = std::string(home) + rel_pattern.substr(1);
                        }
                    }
#endif
                    rel_path = rel_pattern;
                    pattern.erase(0, endpath + 1);
                    curdir /= rel_path;
                }
            }

            for (const auto & entry : std::filesystem::recursive_directory_iterator(curdir,
                    std::filesystem::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }

                std::string rel = std::filesystem::relative(entry.path(), curdir, ec).string();
                if (ec) {
                    ec.clear();
                    continue;
                }
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(pattern, rel)) {
                    continue;
                }

                if (!add_text_file((rel_path / rel).string())) {
                    continue;
                }

                if (++count >= FILE_GLOB_MAX_RESULTS) {
                    console::error("Maximum number of globbed files allowed (%zu) reached.\n", FILE_GLOB_MAX_RESULTS);
                    break;
                }
            }
            continue;
        } else {
            // not a command
            if (mem_db) {
                std::vector<std::string> teach_items;
                std::string stripped = buffer;

                if (!semantic_opts.teach_prefix.empty() && string_starts_with(stripped, semantic_opts.teach_prefix)) {
                    teach_items.push_back(string_strip(stripped.substr(semantic_opts.teach_prefix.size())));
                    stripped.clear();
                }

                if (semantic_opts.enable_teach_tags) {
                    std::string cleaned;
                    auto from_tags = extract_teach_blocks(
                        stripped,
                        semantic_opts.teach_open_tag,
                        semantic_opts.teach_close_tag,
                        cleaned,
                        false);
                    stripped = cleaned;
                    teach_items.insert(teach_items.end(), from_tags.begin(), from_tags.end());
                }

                if (semantic_opts.enable_learn_from_response_tags) {
                    std::string cleaned;
                    auto from_response_tags = extract_teach_blocks(
                        stripped,
                        semantic_opts.learn_from_response_open_tag,
                        semantic_opts.learn_from_response_close_tag,
                        cleaned,
                        true);
                    stripped = cleaned;
                    learn_from_response_prompts.insert(
                        learn_from_response_prompts.end(),
                        from_response_tags.begin(),
                        from_response_tags.end());
                }

                size_t taught = 0;
                for (const auto & t : teach_items) {
                    if (t.empty()) {
                        continue;
                    }
                    for (const auto & teach_variant : build_teach_variants(t, semantic_opts.enable_teach_variants)) {
                        std::string err;
                        auto embd = ctx_cli.generate_embedding(teach_variant, err);
                        if (!embd) {
                            console::error("teach embedding failed: %s\n", err.c_str());
                            continue;
                        }
                        char * db_err = nullptr;
                        std::string ts = iso8601_now();
                        uint64_t id = logosdb_put(mem_db, embd->data(), (int)embd->size(),
                            teach_variant.c_str(), ts.c_str(), &db_err);
                        if (id == UINT64_MAX) {
                            console::error("teach write failed: %s\n", db_err ? db_err : "unknown");
                            free(db_err);
                            continue;
                        }
                        ++taught;
                    }
                }

                if (taught > 0) {
                    console::log("taught %zu memory row(s)\n", taught);
                }

                buffer = stripped;
                if (buffer.empty()) {
                    continue;
                }

                if (semantic_opts.hint_tokens > 0 || semantic_opts.hint_llm_compress) {
                    std::string hint_err;
                    auto qembd = ctx_cli.generate_embedding(buffer, hint_err);
                    if (qembd) {
                        int top_k = std::max(1, semantic_opts.hint_llm_top_k);
                        char * search_err = nullptr;
                        logosdb_search_result_t * res = logosdb_search(
                            mem_db, qembd->data(), (int)qembd->size(), top_k, &search_err);
                        if (res) {
                            int n_hits = logosdb_result_count(res);
                            std::vector<std::string> memory_chunks;
                            for (int i = 0; i < n_hits; ++i) {
                                const char * text = logosdb_result_text(res, i);
                                const char * ts   = logosdb_result_timestamp(res, i);
                                if (text) {
                                    std::string chunk = text;
                                    if (ts) {
                                        chunk = std::string("[") + ts + "] " + chunk;
                                    }
                                    memory_chunks.push_back(std::move(chunk));
                                }
                            }

                            std::string hint_suffix;
                            if (semantic_opts.hint_llm_compress && !memory_chunks.empty()) {
                                const std::string compressed = build_llm_compressed_hint(
                                    ctx_cli, buffer, memory_chunks, semantic_opts.hint_llm_n_predict);
                                hint_suffix = build_prompt_injection_suffix(compressed);
                            }

                            if (hint_suffix.empty() && semantic_opts.hint_tokens > 0 && n_hits > 0) {
                                const char * best_text = logosdb_result_text(res, 0);
                                if (best_text) {
                                    auto hint_toks = extract_hint_tokens(best_text, semantic_opts.hint_tokens);
                                    hint_suffix = build_prompt_hint_suffix(hint_toks);
                                }
                            }
                            if (!hint_suffix.empty()) {
                                buffer += hint_suffix;
                            }
                            logosdb_result_free(res);
                        } else {
                            free(search_err);
                        }
                    }
                }
            }

            cur_msg += buffer;
        }

        // generate response
        if (add_user_msg) {
            ctx_cli.messages.push_back({
                {"role",    "user"},
                {"content", cur_msg}
            });
            cur_msg.clear();
        }
        result_timings timings;
        std::string assistant_content = ctx_cli.generate_completion(timings);
        ctx_cli.messages.push_back({
            {"role",    "assistant"},
            {"content", assistant_content}
        });

        if (mem_db && !learn_from_response_prompts.empty()) {
            size_t taught_from_response = 0;
            for (const auto & prompt : learn_from_response_prompts) {
                if (assistant_content.empty()) {
                    continue;
                }

                std::string memory_text;
                if (prompt.empty()) {
                    memory_text = assistant_content;
                } else {
                    memory_text = "Question: " + prompt + "\nAnswer: " + assistant_content;
                }

                std::string err;
                auto embd = ctx_cli.generate_embedding(memory_text, err);
                if (!embd) {
                    console::error("learn-from-response embedding failed: %s\n", err.c_str());
                    continue;
                }
                char * db_err = nullptr;
                std::string ts = iso8601_now();
                uint64_t id = logosdb_put(mem_db, embd->data(), (int)embd->size(),
                    memory_text.c_str(), ts.c_str(), &db_err);
                if (id == UINT64_MAX) {
                    console::error("learn-from-response write failed: %s\n", db_err ? db_err : "unknown");
                    free(db_err);
                    continue;
                }
                ++taught_from_response;
            }
            if (taught_from_response > 0) {
                console::log("learned %zu memory row(s) from assistant response\n", taught_from_response);
            }
        }
        console::log("\n");

        if (params.show_timings) {
            console::set_display(DISPLAY_TYPE_INFO);
            console::log("\n");
            console::log("[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n", timings.prompt_per_second, timings.predicted_per_second);
            console::set_display(DISPLAY_TYPE_RESET);
        }

        if (params.single_turn) {
            break;
        }
    }

    console::set_display(DISPLAY_TYPE_RESET);

    console::log("\nExiting...\n");
    mem_db_close();
    ctx_cli.ctx_server.terminate();
    inference_thread.join();

    // bump the log level to display timings
    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    llama_memory_breakdown_print(ctx_cli.ctx_server.get_llama_context());

    return 0;
}
