# Portfolio Project: Indian Automatic Number Plate Recognition (ANPR) Studio

**Author:** Biswaranjan Swain  
**GitHub:** [@brslabs786](https://github.com/brslabs786)  
**Repository:** [https://github.com/brslabs786/Numberplate-Recognition-system-](https://github.com/brslabs786/Numberplate-Recognition-system-)  
**Copyright:** © 2026 Biswaranjan Swain. All Rights Reserved.

---

## 📌 Project Summary for Resume / Portfolio

> Developed an end-to-end Computer Vision & Deep Learning web application for **Automatic Number Plate Recognition (ANPR)** optimized for **Indian High Security Registration Plates (HSRP)** and **Bharat (BH) Series** using **YOLOv8**, **EasyOCR**, and **FastAPI**.

### 🌟 Key Highlights & Engineering Achievements:
1. **Dual YOLOv8 Model Pipeline:** Integrated pre-trained COCO object detector for multi-class vehicle identification (Cars, Motorcycles, Trucks, Buses) and custom-trained YOLOv8 for precise license plate localization.
2. **Context-Aware Character Disambiguation Engine:** Engineered an intelligent rule-based post-processor to eliminate classic OCR optical misclassifications (resolving `T` $\leftrightarrow$ `1`, `Z` $\leftrightarrow$ `2`, `O` $\leftrightarrow$ `0`, and `B` $\leftrightarrow$ `8` based on Indian RTO plate grammar).
3. **Multi-Frame Majority Voting Consensus:** Built a multi-frame temporal voting engine for traffic video tracking with SORT, aggregating candidate OCR readings across frames to filter noise and lock onto the highest-resolution consensus reading.
4. **Distance Gate (Minimum Size Filter):** Designed an adaptive resolution filter to prevent far-away background noise from hallucinating incorrect license plate numbers.
5. **Fullstack Production UI:** Developed a glassmorphic dark-mode web studio using **FastAPI**, **Tailwind CSS**, and **Lucide Icons** featuring real-time image & video processing, live scoreboard metrics, realistic HSRP badge rendering, and an interactive **Live JSON Viewer Modal** with 1-click clipboard copy.

---

## 💻 Embeddable Portfolio Card (HTML / Tailwind CSS)

Copy and paste this card into your personal portfolio website:

```html
<!-- ANPR Project Card -->
<div class="rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-xl hover:border-indigo-500/50 transition duration-300 flex flex-col justify-between">
  <div>
    <div class="flex items-center justify-between mb-4">
      <span class="px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
        Computer Vision / AI
      </span>
      <span class="text-xs text-slate-500">2026</span>
    </div>
    
    <h3 class="text-xl font-bold text-white mb-2">AI Indian ANPR & Traffic Studio</h3>
    <p class="text-sm text-slate-400 mb-4 leading-relaxed">
      Fullstack Indian Number Plate Recognition system supporting HSRP & Bharat Series with YOLOv8, EasyOCR, SORT tracking, and real-time FastAPI dashboard.
    </p>

    <!-- Tech Stack Tags -->
    <div class="flex flex-wrap gap-1.5 mb-6">
      <span class="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">Python</span>
      <span class="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">YOLOv8</span>
      <span class="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">EasyOCR</span>
      <span class="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">FastAPI</span>
      <span class="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">Tailwind CSS</span>
    </div>
  </div>

  <div class="flex items-center justify-between pt-4 border-t border-slate-800 text-xs">
    <a href="https://github.com/brslabs786/Numberplate-Recognition-system-" target="_blank" class="text-indigo-400 hover:text-indigo-300 font-semibold flex items-center gap-1.5">
      View Source Code &rarr;
    </a>
    <span class="text-slate-500">© Biswaranjan Swain</span>
  </div>
</div>
```
