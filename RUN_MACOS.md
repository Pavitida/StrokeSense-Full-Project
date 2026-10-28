# ใช้ StrokeSense บน macOS

คู่มือนี้อิงโค้ดในโปรเจกต์ ใช้ Python 3.11 และ CPU เป็นค่าเริ่มต้น ยังไม่ได้ทดสอบติดตั้งหรือกล้อง/ไมโครโฟนบน Mac จริง ความพร้อมของแพ็กเกจเวอร์ชันที่ล็อกไว้ต้องตรวจบน Mac ปลายทาง ทั้ง Apple Silicon และ Intel

## เริ่มใช้งานเฉพาะเว็บบน Mac ชิป M (Apple Silicon)

ใช้ขั้นตอนนี้สำหรับ Mac ตระกูล M โดยเปิดโฟลเดอร์เว็บใน VS Code แล้วเลือก Terminal > New Terminal ก่อน ไม่ต้องติดตั้ง CUDA หรือโมเดลบนเครื่องสำหรับรันเฉพาะเว็บ

1. ติดตั้ง Homebrew ตาม [คู่มือทางการ](https://docs.brew.sh/Installation) และทำตามคำสั่งตั้งค่า shell ที่ตัวติดตั้งแสดง Homebrew แบบ native Apple Silicon ใช้ `/opt/homebrew`
2. ใช้ VS Code / Terminal แบบ Apple Silicon ไม่เปิดผ่าน Rosetta ตรวจคำสั่งด้านล่าง ผลทั้งสองต้องเป็น `arm64` หากเป็น `x86_64` ให้กลับมาเปิด Terminal แบบ native และสร้าง environment ใหม่ อย่าใช้ environment ที่คัดลอกจาก Windows
3. รันคำสั่งทีละบรรทัดจากโฟลเดอร์ที่มี `app.py`:

```bash
uname -m
brew install python@3.11
"$(brew --prefix python@3.11)/bin/python3.11" -c "import platform; print(platform.machine())"
"$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv-web
.venv-web/bin/python -m pip install --upgrade pip
.venv-web/bin/python -m pip install 'streamlit==1.66.1' 'requests==2.34.2'
.venv-web/bin/python -m pip check
.venv-web/bin/python -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --server.maxUploadSize 20 --browser.gatherUsageStats false
```

เปิด **http://localhost:8501** และหยุดด้วย **Ctrl+C** ครั้งถัดไปใช้เพียงคำสั่งเปิด Streamlit บรรทัดสุดท้าย ไม่ต้องติดตั้งซ้ำ เลือก interpreter `.venv-web/bin/python` ใน VS Code ผ่าน Python: Select Interpreter

หาก `brew: command not found` ทั้งที่ติดตั้ง Homebrew แล้ว ให้เปิด Terminal ใหม่ หรือรัน `eval "$(/opt/homebrew/bin/brew shellenv)"` ใน Terminal แบบ native ก่อนทำต่อ หากติดตั้งแพ็กเกจไม่สำเร็จให้เก็บข้อความ error พร้อมผล `sw_vers` และ `uname -m` เพื่อหาสาเหตุ ไม่เปลี่ยนเวอร์ชันโมเดลโดยเดา

หน้าเว็บเปิดและรับสื่อได้ แต่การวิเคราะห์จริงต้องตั้งค่า Backend ตามหัวข้อด้านล่าง การรันเฉพาะเว็บไม่ได้ใช้ GPU ของชิป M ประมวลผลโมเดล ส่วน `run-macos.sh` สำหรับระบบเต็มตั้งค่า CPU ไว้ ยังไม่ได้ยืนยัน GPU/Metal ของโมเดลบน Mac

อ้างอิงขั้นตอน environment จาก [Streamlit installation](https://docs.streamlit.io/get-started/installation/command-line) ยังไม่ได้ทดสอบติดตั้งแพ็กเกจที่ล็อกไว้หรือบันทึกสื่อบน Mac จริง

## รันเฉพาะหน้าเว็บบน Mac (ไม่ติดตั้งโมเดล)

เปิดโฟลเดอร์โปรเจกต์ใน VS Code ด้วย File > Open Folder แล้วเลือก Terminal > New Terminal ตรวจว่า Terminal อยู่ในโฟลเดอร์ที่มี `app.py` คำสั่งด้านล่างใช้กับ shell zsh/bash บน Mac

ต้องมี `app.py`, `src/capture.py` และ `src/video_recorder.js` อยู่ตามโครงสร้างเดิม ไม่ต้องใช้ `Data/` หรือ `models/` สำหรับเปิดหน้าเว็บ หากโค้ดเวอร์ชันนี้ยังไม่ได้ push ให้คัดลอกไฟล์เหล่านี้จากเครื่องที่มีเว็บล่าสุดก่อน

ติดตั้ง Homebrew จาก https://brew.sh หากยังไม่มี แล้วเตรียม Python และ environment แยกสำหรับเว็บ:

```bash
# ทำเมื่อยังไม่มี Xcode Command Line Tools
xcode-select --install

brew install python@3.11
"$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv-web
.venv-web/bin/python -m pip install --upgrade pip
.venv-web/bin/python -m pip install 'streamlit==1.66.1' 'requests==2.34.2'
.venv-web/bin/python -m pip check
```

เปิดหน้าเว็บเพื่อดูหน้า Home และทดลองการรับภาพ/เสียง/วิดีโอ:

```bash
.venv-web/bin/python -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --server.maxUploadSize 20 --browser.gatherUsageStats false
```

เปิด http://localhost:8501 หยุดด้วย Ctrl+C การเปิดหน้าเว็บอย่างเดียวไม่สามารถวิเคราะห์หรือแสดงผลโมเดลจริงได้ และไม่แสดงผลจำลองแทน

### เชื่อมต่อ Backend ที่มีอยู่แล้ว

หยุดเว็บก่อน แล้วตั้งค่าใน Terminal เดียวกับที่จะเปิดเว็บ เปลี่ยน URL ตัวอย่างให้เป็น API ของทีม ไม่ใช่ URL หน้าเว็บ 8501 และไม่เติม `/api/analyze`:

```bash
export STROKESENSE_API_URL='https://api.example.com'
# รับ token แบบซ่อนอักขระ ใช้ได้ทั้ง zsh และ bash
printf 'Backend API token: '
read -rs STROKESENSE_API_TOKEN
printf '\n'
export STROKESENSE_API_TOKEN

.venv-web/bin/python -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --server.maxUploadSize 20 --browser.gatherUsageStats false

# หลังหยุดเว็บ
unset STROKESENSE_API_TOKEN
```

Token ต้องตรงกับ Backend ที่ให้บริการ API ของ StrokeSense และเก็บเฉพาะฝั่งเซิร์ฟเวอร์ ห้ามใส่ใน Git หรือ JavaScript ฝั่ง browser ตัวอย่าง URL ไม่ใช่บริการที่เปิดไว้แล้ว หากยังไม่มี Backend ให้ใช้ขั้นตอนรันครบทั้งระบบด้านล่าง

`localhost` บน Mac หมายถึง Mac เครื่องนั้น หากโมเดลอยู่บน Windows/Ubuntu อีกเครื่อง ต้องเตรียมช่องทางเชื่อมต่อที่ปลอดภัยก่อน ตัว launcher ปัจจุบันรับเฉพาะ loopback จึงยังเข้าจากเครื่องอื่นโดยตรงไม่ได้

### กล้องและไมค์

อนุญาตสิทธิ์ทั้งใน browser และ System Settings > Privacy & Security > Camera / Microphone หากเคยปฏิเสธให้แก้ site settings แล้วโหลดหน้าใหม่ หากยังบันทึกไม่ได้ให้ใช้การอัปโหลดไฟล์ Safari และอุปกรณ์ Mac จริงยังไม่ได้ทดสอบกับเว็บนี้

แนวทางสร้าง environment และเปิด Streamlit อ้างอิง [Streamlit installation](https://docs.streamlit.io/get-started/installation/command-line) และ [Homebrew Python 3.11](https://formulae.brew.sh/formula/python@3.11) ส่วนเวอร์ชันที่ระบุยึดตามโปรเจกต์ ยังไม่ได้ตรวจการติดตั้งบน Mac จริง

---

ขั้นตอนต่อไปนี้สำหรับรันทั้งเว็บและโมเดลบน Mac เครื่องเดียวกัน

## 1. เตรียมเครื่อง

ติดตั้ง Homebrew ตาม https://brew.sh หากยังไม่มี แล้วเปิด Terminal:

```bash
brew install python@3.11 git
```

บน Apple Silicon ให้ใช้ Terminal และ Python แบบ native arm64 ตรวจด้วย:

```bash
uname -m
"$(brew --prefix python@3.11)/bin/python3.11" -c "import platform; print(platform.machine())"
```

ค่าควรตรงกัน: arm64 สำหรับ Apple Silicon หรือ x86_64 สำหรับ Intel อย่าคัดลอก `.venv`, `.venv-ubuntu`, `.python` จาก Windows/Ubuntu มาใช้บน Mac

## 2. เตรียมโค้ดและ environment

หากยังไม่มีโฟลเดอร์โปรเจกต์:

```bash
git clone https://github.com/Pavitida/StrokeSense-Full-Project.git
cd StrokeSense-Full-Project
```

ถ้ามีโฟลเดอร์อยู่แล้ว ให้ `cd` เข้าโฟลเดอร์นั้นแทน ต้องเห็น `app.py`, `src/` และ `scripts/` คู่มือนี้กับ `requirements-macos.txt` และ `run-macos.sh` ต้องอยู่ที่ root ของโปรเจกต์ด้วย หากยังไม่ได้ push ไฟล์เหล่านี้ขึ้น GitHub ให้คัดลอกไปพร้อมโค้ด

```bash
"$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-macos.txt
.venv/bin/python -m pip check
```

`requirements-macos.txt` ใช้รายการเวอร์ชันเดียวกับ `requirements-ubuntu.txt` ซึ่งเป็น Python dependencies ไม่ได้รัน apt หรือใช้ WSL ไม่ใช่ lockfile ที่ยืนยันบน Mac แล้ว หากพบ `No matching distribution found` ให้เก็บชื่อแพ็กเกจ เวอร์ชัน macOS และผล `uname -m` เพื่อตรวจ wheel อย่าปลด version pin ทั้งหมด เพราะโมเดล pickle ต้องใช้ไลบรารีที่เข้ากันได้

ตรวจ import:

```bash
.venv/bin/python -c "import numpy, pandas, sklearn, joblib, cv2, mediapipe, librosa, soundfile, streamlit, starlette, uvicorn, multipart, requests; print('Imports OK')"
```

ตรวจว่ามีไฟล์โมเดลเหล่านี้ และเป็นไฟล์จริง ไม่ใช่ Git LFS pointer:

- `models/face_landmark_model_v2.pkl`
- `models/speech_model.pkl`
- `models/movement_model_final.pkl`
- `models/mediapipe/face_landmarker.task`
- `models/mediapipe/pose_landmarker.task`

## 3. เปิดแอป

```bash
bash run-macos.sh
```

หรือใช้คำสั่งเดียวกันโดยตรง:

```bash
STROKESENSE_DEVICE=cpu .venv/bin/python scripts/serve.py
```

เปิด http://localhost:8501 และหยุดด้วย Ctrl+C ตัว launcher เปิดทั้ง API และ Streamlit พร้อม token ภายใน ไม่ควรเปิดเฉพาะ `streamlit run app.py` ไม่ต้องใช้ `run-ubuntu.sh`, WSL, CUDA หรือค่า NVIDIA/Mesa บน Mac

อนุญาตกล้องและไมโครโฟนให้ browser ทั้งใน site settings และ macOS System Settings > Privacy & Security หากบันทึกไม่ได้ ใช้อัปโหลด JPEG/PNG, WAV และ MP4/MOV/WebM แทน Safari และฮาร์ดแวร์ Mac ยังไม่ได้ตรวจจริงกับโปรเจกต์นี้

หาก port 8501/8502 ถูกใช้ ให้หยุด StrokeSense instance เดิมก่อน

## 4. ตรวจโมเดลแยกส่วน

เปลี่ยน path ตัวอย่างเป็นไฟล์จริงของคุณ:

```bash
STROKESENSE_DEVICE=cpu .venv/bin/python -m src.face.face_inference --image "/path/to/face.jpg"
STROKESENSE_DEVICE=cpu .venv/bin/python -m src.speech.speech_inference --audio "/path/to/speech.wav"
STROKESENSE_DEVICE=cpu .venv/bin/python -m src.movement.movement_inference --video "/path/to/movement.mp4"
```

## 5. เทรน Fusion เมื่อข้อมูลพร้อม

ยังไม่พบข้อมูลจับคู่ในเครื่อง Windows ที่ตรวจ จึงยังไม่ได้เทรน ขั้นตอนนี้ใช้เมื่อมีไฟล์จริงแล้วเท่านั้น

วาง CSV ที่ `data/fusion/paired_fusion_data.csv` โดยมี header:

```csv
participant_id,face_score,speech_score,movement_score,label
```

แต่ละแถวต้องเป็นคนเดียวกันในการประเมินครั้งเดียวกัน คะแนนทั้งสามอยู่ระหว่าง 0–1 และ label เป็นผลอ้างอิงที่นิยามชัดเจน มีทั้ง 0 และ 1 ห้ามนำข้อมูลคนละคนมาจับคู่หรือสร้าง label จากคะแนนโมเดลเอง หากได้เพียงหน้าและเสียงยังไม่ครบสำหรับ trainer สามส่วนนี้

```bash
.venv/bin/python -m src.fusion.train_fusion \
  --data data/fusion/paired_fusion_data.csv \
  --output models/fusion_model.pkl
```

คำสั่งนี้เขียนทับไฟล์ปลายทางถ้ามีอยู่แล้ว หากต้องการเก็บโมเดลเดิมให้ใช้ชื่อ output ใหม่ สคริปต์แบ่ง train/test ตาม participant_id ต้องตรวจว่าทั้งสองส่วนมีทั้งสองคลาส และผู้เข้าร่วมในชุดทดสอบไม่ได้อยู่ในชุดฝึกโมเดลย่อยด้วย ก่อนตีความผลประเมิน

หลังเทรนสำเร็จ ใช้ไฟล์ที่ `models/fusion_model.pkl` แล้วหยุดและเปิดแอปใหม่ Fusion จะคำนวณเมื่อทั้งสามโมดูลสำเร็จ หากยังไม่มีโมเดล ผลรวมเป็น `incomplete` ตามการออกแบบ คะแนน Movement เป็นการจำแนกความสมบูรณ์ของท่าฟื้นฟู คะแนน Fusion ยังไม่ใช่ความน่าจะเป็น Stroke ที่ผ่านการตรวจสอบทางคลินิก

อย่า commit ข้อมูลผู้เข้าร่วมลง Git ตรวจชื่อ `Data/` กับ `data/` ให้ตรงตามโค้ด โดยเฉพาะ Mac ที่ใช้ filesystem แบบ case-sensitive

## ขอบเขตการตรวจและแหล่งอ้างอิง

ตรวจคำสั่งกับ entry point ของโค้ดแล้ว แต่ยังไม่ได้ยืนยันการติดตั้ง dependency, โหลดโมเดล หรือใช้กล้องบน macOS จริง การ import ผ่านอย่างเดียวไม่ยืนยันความแม่นยำของโมเดล

- Homebrew Python 3.11: https://formulae.brew.sh/formula/python@3.11
- MediaPipe Python setup: https://developers.google.com/edge/mediapipe/solutions/setup_python
