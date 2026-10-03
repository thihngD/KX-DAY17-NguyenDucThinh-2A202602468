# Bước 8 — Phân tích kết quả benchmark (Day 17: Memory Systems)

Mọi con số trong file này lấy từ đúng lệnh `python src/benchmark.py` (chế độ offline, tất định) và vài script đo phụ chạy trên chính code trong `src/`. Không có số nào được ước lượng tay.

## Cấu hình khi đo

| Tham số | Giá trị | Nằm ở |
|---|---|---|
| `compact_threshold_tokens` | 600 | `src/config.py` (`COMPACT_THRESHOLD_TOKENS`) |
| `compact_keep_messages` | 4 | `src/config.py` (`COMPACT_KEEP_MESSAGES`) |
| `profile_confidence_threshold` | 0.6 | `src/config.py` (`PROFILE_CONFIDENCE_THRESHOLD`) |
| Token estimator | `len(text) // 4` | `src/memory_store.py::estimate_tokens` |

Cách đếm token trong benchmark (`src/benchmark.py::run_agent_benchmark`):

- `Agent tokens only`: tổng `estimate_tokens(reply)` của mọi lượt, gồm cả thread chat lẫn thread hỏi recall.
- `Prompt tokens processed`: tổng ngữ cảnh mà agent mang theo ở **từng lượt**, cộng dồn ngay trong `reply()`.
  - Baseline: toàn bộ message của thread.
  - Advanced: `User.md` + summary + các message được giữ lại.
- `recall_questions` được hỏi ở thread `<conv-id>::recall`, là thread mới và đặt tên giống nhau cho cả hai agent.
- Mỗi suite chạy với thư mục `state/benchmark/<suite>/` được xóa trước khi chạy, nên kết quả không phụ thuộc lần chạy trước.

## Kết quả

### Standard Benchmark (`data/conversations.json`)

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                 585 |                      9641 |                      0 |                0.2 |                       0 |             0 |
| Advanced |                 583 |                     17896 |                      1 |                1   |                     370 |             0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                 180 |                     21213 |                      0 |                0.2 |                       0 |             0 |
| Advanced |                 224 |                      7976 |                      1 |                1   |                     273 |             7 |

Ghi chú về cột `Response quality`: công thức là `0.8 × độ phủ fact + 0.2 × độ ngắn` (`heuristic_quality`) và áp dụng giống nhau cho cả hai agent. Baseline được 0.20 hoàn toàn nhờ phần "ngắn"; độ phủ fact của Baseline bằng 0.

---

## 1. Vì sao Advanced có recall tốt hơn Baseline?

**Số liệu.** Cross-session recall của Advanced là 1.00 ở cả hai bảng, của Baseline là 0 ở cả hai bảng. Memory growth của Baseline bằng 0 byte vì Baseline không ghi file nào. Advanced ghi 370 byte (Standard) và 273 byte (Stress) vào `User.md`.

**Cơ chế.** Ở mỗi lượt, `AdvancedAgent._remember()` gọi `extract_profile_updates()` để trích fact. Sau đó `UserProfileStore.apply_updates()` ghi fact thành dòng `- key: value` trong `state/.../profiles/<user>/User.md`. Sang thread recall mới, `_offline_response()` đọc lại `profile_store.facts(user_id)` để trả lời.

Baseline lưu `SessionState` theo `thread_id`. Thread `::recall` vì vậy bắt đầu rỗng và trả lời "chưa có thông tin". Trong cùng một thread thì Baseline vẫn nhớ được: `test_cross_session_recall` kiểm tra Baseline trả lời đúng tên trong `session-1` và không biết gì ở `session-2`.

**Phần nào thực sự làm recall đúng.** Mình chạy ablation: tắt bộ lọc phủ định (`_NEGATION`) trong extractor, giữ nguyên mọi thứ khác. Recall Standard của Advanced giảm từ **1.00 xuống 0.86**. Ba câu sai:

- conv-03 trả lời "Đà Nẵng". Nguyên nhân là câu "đang ở Huế chứ **không còn** ở Đà Nẵng" bị đọc thành Đà Nẵng.
- conv-09 và conv-10 trả lời "backend engineer". Nguyên nhân là câu "chứ **không còn** là backend engineer" ghi đè MLOps.

Như vậy có `User.md` mới chỉ là điều kiện cần. Recall đúng phụ thuộc vào việc lưu đúng fact mới nhất.

**Giới hạn.** Các regex và từ điển (thành phố, nghề, keyword style) được thiết kế khi đã nhìn thấy chính bộ dữ liệu này. Vì vậy recall 1.00 là con số lạc quan cho dữ liệu đã biết. Mình chưa đo khả năng tổng quát hóa, ví dụ một thành phố nằm ngoài danh sách `LOCATIONS` sẽ không bao giờ được lưu.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

**Số liệu (bảng Standard).**

- `Prompt tokens processed`: Advanced 17896 so với Baseline 9641, tức Advanced tốn gấp khoảng 1.86 lần.
- `Agent tokens only` gần như bằng nhau: 583 so với 585.
- `Compactions` của Advanced bằng 0.
- Advanced tốn hơn Baseline ở **cả 10/10** hội thoại. Ví dụ conv-01 là 1152 so với 735, conv-10 là 2173 so với 1282 (đo theo từng thread chat).

**Cơ chế.** `_estimate_prompt_context_tokens()` cộng `estimate_tokens(User.md)` vào **mọi lượt**. Hội thoại Standard chỉ khoảng 10 lượt ngắn, tổng ngữ cảnh luôn dưới ngưỡng 600 token, nên `CompactMemoryManager` không compact lần nào (0 compaction). Kết quả là Advanced mang toàn bộ lịch sử thread giống hệt Baseline, cộng thêm `User.md`. Khoảng chênh xấp xỉ bằng (số lượt) × (kích thước `User.md`), và tăng dần khi `User.md` lớn lên từ 244 byte (sau conv-01) lên 370 byte (từ conv-05).

`Agent tokens only` không chênh vì hai agent dùng cùng câu xác nhận "Đã ghi nhận." và cùng định dạng bullet khi trả lời recall. Baseline in "chưa có thông tin", còn Advanced in giá trị thật; độ dài hai kiểu này xấp xỉ nhau.

**Giới hạn.** Chi phí trả thêm này là để mua recall: Advanced đạt 1.00, Baseline đạt 0. Ở hội thoại ngắn, compact chưa có gì để bù lại phần chi phí `User.md`.

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

**Số liệu (bảng Stress).** `Prompt tokens processed` của Advanced là **7976**, của Baseline là **21213**, tức Advanced giảm khoảng **62%**, với 7 lần compaction. Cột `Agent tokens only` thì **không** giảm: Advanced 224 so với Baseline 180, tức cao hơn khoảng 24%. Phần cao hơn đến từ việc Advanced trả lời bằng fact thật (danh sách style dài) thay vì "chưa có thông tin".

Prompt token theo từng lượt (đo bằng script trên cùng code):

| Lượt | 1 | 3 | 4 | 6 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| Baseline | 186 | 492 | 651 | 970 | 1385 | 1829 | 2382 |
| Advanced | 225 | 531 | 386 | 432 | 569 | 434 | 415 |
| Số compaction của Advanced | 0 | 0 | 1 | 2 | 3 | 5 | 7 |

**Cơ chế.**

- Baseline gửi lại toàn bộ thread ở mỗi lượt, nên prompt mỗi lượt tăng tuyến tính và tổng prompt tăng theo bậc hai theo số lượt.
- Ở Advanced, khi `summary + messages` vượt 600 token, `CompactMemoryManager._compact()` gấp các message cũ thành tối đa 6 dòng tóm tắt (mỗi dòng ≤ 100 ký tự) và chỉ giữ 4 message gần nhất. Vì vậy từ lượt 4 trở đi, prompt mỗi lượt của Advanced dao động quanh 386–592 thay vì tăng mãi.
- Ba lượt đầu, Advanced vẫn đắt hơn Baseline, vì lý do giống câu 2.

Compact tối ưu **lượng ngữ cảnh đưa vào** (`Prompt tokens processed`), không tối ưu **lượng token agent sinh ra** (`Agent tokens only`). Hai cột này phải đọc tách riêng.

**Giới hạn.** Summary là heuristic: lấy câu đầu của mỗi message rồi cắt ngắn, và chỉ giữ 6 dòng gần nhất. Vì vậy nội dung chi tiết của bốn bản tin (Artemis III, X-59, WMO, BC) bị mất dần qua các lần compact. Recall vẫn đạt 1.00 chỉ vì cả 3 câu hỏi recall đều hỏi fact ổn định nằm trong `User.md`. Benchmark này **không** đo được việc summary làm mất ngữ cảnh hội thoại.

## 4. File memory tăng trưởng ra sao và rủi ro gì?

**Số liệu.**

- Stress: Memory growth 273 byte, Compactions 7. Theo từng lượt, `User.md` tăng 166 → 235 → 252 → 273 byte ở các lượt 1, 7, 8, 9, rồi **đứng yên ở 273 byte trong 7 lượt cuối**, dù các lượt đó rất dài.
- Standard: `User.md` tăng 244 → 259 → 294 → 345 → 372 byte qua conv-01 đến conv-05, rồi **giảm còn 370 byte ở conv-06** và giữ nguyên đến conv-10.

**Cơ chế.**

- `upsert_fact()` thay thế dòng `- key: value` cũ, không ghi thêm dòng mới. Vì vậy đính chính không làm file to ra.
  - conv-06: "backend engineer" được thay bằng "MLOps engineer", chuỗi mới ngắn hơn 2 ký tự nên file giảm 2 byte.
  - Stress t9: Huế được thay bằng Đà Nẵng.
- Tin tức dài không đi vào `User.md` vì extractor chỉ lưu các field có schema cố định. Phần đó chỉ nằm trong compact memory.

**Rủi ro quan sát được.**

1. **Field dạng list chỉ tăng, không bao giờ giảm.** `response_style` của user `dungct` có 7 mục: "ngắn gọn, rõ ý, ví dụ thực tế, bullet ngắn, ví dụ thực chiến, trade-off, có cấu trúc". Trong đó có các mục gần trùng nhau ("ngắn gọn"/"bullet ngắn", "ví dụ thực tế"/"ví dụ thực chiến"). Nếu chạy nhiều phiên hơn, file sẽ phình đúng ở chỗ này. Hiện chưa có memory decay.
2. **Lưu sai fact khi hiểu sai câu.** Ablation ở câu 1 cho thấy nếu bộ lọc phủ định sai, fact cũ sai ("backend engineer", "Đà Nẵng") sẽ được giữ lại qua mọi phiên sau. Lỗi này bền vững, không tự hết như lỗi trong short-term memory.
3. **Bỏ sót fact thật** (xem phần bonus): conv-08 t8 "mình đang ở Huế để dùng ví dụ địa phương nếu cần" bị bỏ qua vì có chữ "nếu".
4. **Giá trị thô.** `pet` được lưu là "corgi tên Bơ" chứ không phải `{loài: corgi, tên: Bơ}`.
5. **Không giới hạn kích thước.** `User.md` được nạp nguyên văn vào mọi lượt. Với 370 byte thì chưa sao, nhưng đây chính là phần chi phí tạo ra khoảng chênh ở câu 2. File càng lớn thì chi phí cố định mỗi lượt càng tăng.

---

## Bonus: Conflict handling + Confidence threshold + Entity extraction có cấu trúc

Mã nguồn: `src/memory_store.py` (`extract_profile_candidates`, `extract_profile_updates`, `UserProfileStore.apply_updates`).

### Bonus giải quyết vấn đề gì

- **Entity extraction có cấu trúc.** `User.md` có schema cố định gồm 8 field (`PROFILE_FIELDS`). Field scalar (name, location, profession, …) chỉ có một giá trị. Field list (interests, response_style) được hợp nhất.
- **Conflict handling.**
  - Một nơi ở hoặc nghề đứng sau "không còn / không phải / chứ không / đừng / chưa" bị loại khỏi danh sách ứng viên.
  - Với field scalar, đề cập hợp lệ cuối cùng trong message thắng ("từ Huế **sang** Đà Nẵng").
  - `upsert_fact()` ghi đè thay vì thêm dòng, nên `User.md` không giữ song song fact cũ đã sai.
- **Guard câu hỏi.** Câu có "?" hoặc kết thúc bằng gì/không/đâu/nào/ai/sao bị bỏ qua. Ví dụ "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì." không bị lưu thành `favorite_drink: gì`; `test_user_markdown_read_write_edit` kiểm tra điều này.
- **Confidence threshold.**
  - Fact có trigger rõ ràng nhận confidence 0.9.
  - Fact nằm trong câu giả định hoặc đùa ("nếu", "đùa", "giả sử", "hay là", "ví dụ cũ") nhận 0.3.
  - Chỉ fact có confidence ≥ 0.6 mới được ghi.

### Cải thiện recall / token thế nào (có đo)

- **Conflict handling:** tắt đi thì recall Standard giảm 1.00 → 0.86 (3/14 câu sai, chi tiết ở câu 1). Đây là phần bonus có tác động đo được lớn nhất.
- **Confidence threshold:** chạy lại benchmark với `PROFILE_CONFIDENCE_THRESHOLD=0` cho **hai bảng giống hệt** bảng mặc định. Trên bộ dữ liệu này, threshold **không** cải thiện recall. Lý do là trong stress t10, câu đùa "hay là chuyển sang product manager" (confidence 0.3) nằm cùng message với "Nghề nghiệp hiện tại vẫn là MLOps engineer" (0.9), nên quy tắc "đề cập cuối thắng" đã xử lý được.
  - Threshold chỉ có tác dụng khi câu đùa đứng riêng một lượt. Với message `"Có lúc mình đùa rằng hay là chuyển sang product manager."`, threshold 0.6 cho `{}`, còn threshold 0 cho `{'profession': 'product manager'}`.
  - `test_cross_session_recall` kiểm tra trường hợp này.
- **Token:** bonus không tạo thêm token prompt. Fact được chuẩn hóa thành dòng ngắn, và đính chính thay thế dòng cũ thay vì thêm dòng, nên Memory growth đứng yên sau khi profile ổn định (xem câu 4).

### Rủi ro bonus tạo thêm

1. **False negative do threshold.** So sánh threshold 0 với 0.6 trên toàn bộ 117 lượt chat của hai dataset (101 lượt Standard + 16 lượt Stress), chỉ có **một** khác biệt: conv-08 t8 "Bạn nhớ là mình đang ở Huế để dùng ví dụ địa phương **nếu** cần" bị bỏ vì chữ "nếu" nằm ở mệnh đề phụ. Recall không bị ảnh hưởng vì Huế đã được lưu từ trước. Nhưng nếu đây là lần đầu user nói nơi ở thì fact đúng sẽ bị mất.
2. **Phủ định theo cửa sổ 20 ký tự là heuristic.** Một câu như "mình không nghĩ là sẽ rời Huế" có thể bị hiểu sai. Lỗi kiểu này **bền vững** trong `User.md`, nặng hơn lỗi short-term.
3. **"Đề cập cuối thắng" có thể sai** khi user kể lại quá khứ ở cuối câu, ví dụ "mình ở Đà Nẵng, trước đây ở Huế": "trước đây" chưa được coi là phủ định.
4. **Phụ thuộc từ điển.** `LOCATIONS`, `_PROFESSION_RE`, `STYLE_KEYWORDS` cố định, nên giá trị ngoài danh sách bị bỏ qua hoàn toàn. Đổi lại, extractor không lưu rác.
5. **Độ phức tạp.** Logic trích fact giờ có nhiều quy tắc tương tác với nhau (câu hỏi → phủ định → hedge → thứ tự). Mỗi quy tắc mới cần test riêng, nếu không sẽ dễ phá recall ở chỗ khác.

## Tóm tắt luồng logic theo Rubric

1. **Baseline không nhớ dài hạn:** recall 0 ở cả hai bảng, memory growth 0 byte.
2. **Advanced thêm `User.md` nên recall tăng:** recall 1.00 ở cả hai bảng. Fact đính chính được giữ đúng nhờ conflict handling (ablation 1.00 → 0.86).
3. **Hội thoại dài làm prompt cost tăng mạnh:** prompt mỗi lượt của Baseline tăng 186 → 2382 token, tổng 21213.
4. **Compact kéo chi phí ngữ cảnh xuống:** Advanced còn 7976 token (−62%) với 7 compaction. `Agent tokens only` không giảm (224 so với 180). Ở hội thoại ngắn, compact không kích hoạt và Advanced tốn gấp 1.86 lần.
5. **Hệ thống mạnh hơn nhưng phức tạp hơn và cần guardrail:** cần guard câu hỏi, phủ định, threshold. Mỗi guardrail có cái giá riêng (false negative ở conv-08, list style phình 7 mục, summary làm mất chi tiết tin tức).

## Phụ lục: kết quả chế độ live (`gpt-4o-mini`)

Toàn bộ phân tích ở trên dùng số liệu **offline**: tất định, chạy lại cho đúng cùng một số.

Phần này là kết quả của lệnh `python src/benchmark.py --live`:
- Model: `gpt-4o-mini` qua OpenAI, `temperature=0`. Ngưỡng compact và confidence giữ nguyên như bản offline.
- Cách đếm token: lấy `usage_metadata` thật do API trả về (`input_tokens` / `output_tokens`), không dùng `len // 4`.
- Mỗi cấu hình chỉ chạy **một lần**. LLM không tất định nên chạy lại có thể ra số khác.

### Standard Benchmark (live)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline |              7313 |                   48540 |                 0.07 |             0.23 |                     0 |           0 |
| Advanced |              7474 |                  116034 |                 0.96 |             0.97 |                   687 |           2 |

### Long-Context Stress Benchmark (live)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline |              3960 |                   51225 |                    0 |             0.19 |                     0 |           0 |
| Advanced |              3788 |                   33212 |                    1 |             1    |                   604 |          27 |

### Số live có giữ được kết luận của bản offline không

- **Câu 1 (recall):** có. Recall của Advanced là 0.96 và 1.00, của Baseline là 0.07 và 0.
  - Baseline có 0.07 vì ở conv-05 và conv-09 model lặp lại chữ "DũngCT" có sẵn trong câu hỏi, ví dụ: "Chưa có thông tin về DũngCT trong cuộc trò chuyện này."
  - Đây là hạn chế của cách chấm `recall_points` bằng so khớp chuỗi, không phải Baseline nhớ được.
- **Câu 2 (hội thoại ngắn):** có, và chênh lệch còn rõ hơn. Prompt tokens của Advanced là 116034, của Baseline là 48540, tức gấp khoảng **2.4 lần** (offline là 1.86 lần).
  - Live đắt hơn vì system prompt của Advanced ngoài `User.md` còn có hướng dẫn và mô tả của 2 tool.
  - Ngoài ra, mỗi lần model gọi tool là thêm một lượt gọi model, và lượt đó phải gửi lại toàn bộ ngữ cảnh.
- **Câu 3 (hội thoại dài):** có. Prompt tokens của Advanced là 33212, của Baseline là 51225, tức giảm khoảng **35%** (offline là 62%), với 27 lần compaction.
  - Mức giảm nhỏ hơn offline vì câu trả lời thật của LLM dài hơn nhiều so với "Đã ghi nhận.". Các câu trả lời này cũng nằm trong 4 message được giữ lại, nên compact phải chạy liên tục (27 lần so với 7 lần offline).
  - `Agent tokens only` gần như không đổi: 3788 so với 3960. Kết luận "compact tối ưu prompt, không tối ưu output" vẫn đúng.
- **Câu 4 (memory growth):** live phình gần gấp đôi offline: **687** so với 370 byte, và **604** so với 273 byte.
  - Toàn bộ phần tăng thêm đến từ tool `save_user_fact`. LLM ghi các "mối quan tâm" như "memory file tăng trưởng theo thời gian" hay "within-session memory", là chủ đề hội thoại chứ không phải sở thích ổn định.
  - Các mục này vẫn qua được guardrail vì mỗi mục ≤ 40 ký tự.
  - Hậu quả đo được: câu recall duy nhất Advanced trả lời thiếu ở bản live (conv-09, recall 0.5) là do model lấy hai mục rác này làm "hai mối quan tâm kỹ thuật chính" thay vì Python và AI.
  - Đây đúng là rủi ro "lưu sai fact" của câu 4, và nó xuất hiện khi giao quyền ghi memory cho LLM.

### Lỗi mà chỉ chế độ live mới phát hiện ra (đã sửa trước khi đo các số trên)

Lần chạy live đầu tiên cho recall Standard của Advanced chỉ **0.89**, và `User.md` của user `dungct` chỉ còn **62 byte**: mất header, mất các fact nghề nghiệp, món ăn và thú cưng.

**Nguyên nhân 1 — ghi đồng thời.** LangGraph chạy các tool call song song trong nhiều thread, nên các lời gọi `save_user_fact` cùng đọc-sửa-ghi một file.
- Cách sửa: thêm `RLock` cho `UserProfileStore` và ghi file atomic (ghi vào file tạm rồi `os.replace`).
- Test `test_user_markdown_read_write_edit` có thêm kịch bản 8 thread ghi cùng lúc. Khi tắt lock, kịch bản này fail **20/20** lần thử.

**Nguyên nhân 2 — LLM ghi đè fact đúng bằng giá trị kém hơn.** Ví dụ "cà phê sữa đá" bị ghi đè thành "cà phê", làm sai câu recall ở conv-08. LLM cũng nhồi nguyên câu vào `interests`, khiến file stress phình lên 1658 byte.
- Cách sửa:
  - Tool chỉ được điền field còn trống (`overwrite_scalars=False`). Việc đính chính vẫn do bộ xử lý conflict dựa trên luật đảm nhận.
  - Mỗi mục trong list tối đa 40 ký tự, mỗi list tối đa 8 mục (giữ các mục mới nhất).
- Sau khi sửa, recall Standard tăng 0.89 → 0.96 và file stress giảm 1658 → 604 byte.

**Bài học:** giao quyền ghi memory cho LLM làm tăng độ linh hoạt nhưng cần thêm guardrail. Kể cả khi đã có guardrail, LLM vẫn lưu "chủ đề đang bàn" thành "sở thích" (xem câu 4 ở trên). Hướng khắc phục tiếp theo có thể là confidence threshold áp dụng riêng cho các lần ghi qua tool, hoặc memory decay cho field list.
