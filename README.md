![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Visual Prompt Injection and Mitigation Strategies

## 📌 Project Summary
This repository contains an implementation of visual prompt injection attacks and an initial defense evaluation as discussed in [this article](https://brave.com/blog/unseeable-prompt-injections/). The work demonstrates how imperceptible text injected into images can manipulate Large Vision-Language Models (LVLMs) and AI-powered browsers to produce malicious outputs or execute unauthorized actions.

Future implementations plan to use systematic adversarial evaluation and defense mechanisms as proposed in  [this paper](https://huggingface.co/papers/2503.11519)

## 🎯 Motivation
> "A whole new paradigm would be needed to solve prompt injections 10/10 times – It may well be that LLMs can never be used for certain purposes. We're working on some new approaches, and it looks like synthetic data will be a key element in preventing prompt injections."  
> — Sam Altman, via Marvin von Hagen  
> [Source](https://simonwillison.net/2023/May/25/sam-altman/)

Thus, as AI systems increasingly integrate multimodal capabilities—especially vision and language—the security risks from visual prompt injections become more urgent. These attacks exploit the inherent trust that models place in visual inputs, allowing malicious actors to bypass traditional text-based safeguards. This project seeks to expose, understand, and eventually mitigate these vulnerabilities to enable safer deployment of vision-language AI in sensitive, real-world applications.

## 🧩 Novelty
- **Visual Stealth Injection**: Unlike traditional text-based prompt injections, this method embeds malicious instructions directly into images using color, saturation, and brightness manipulation to make the text nearly invisible to humans while remaining detectable by AI.
- **Real-World Browser Integration**: Demonstrates the attack in an AI browser context (inspired by Brave's research), showing how screenshots and webpage content can serve as vectors for indirect prompt injection.

## 🧠 Methodology
1. **Dataset**: Uses the [Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) dataset of 9,200+ images as a benign base for injection.
2. **Injection Technique**:  
   - For each image, a variable injection text is defined.  
   - A complementary color to the average color of a random image region is selected.  
   - The text is rendered with **low saturation and high brightness** to blend into the image while remaining machine-readable.  
3. **Evaluation**:  
   - Injects the prompt into each image.  
   - Uses **Gemini 3.0 Flash** to process the image and classify the injection as a **success** (model output aligns with injected text) or **fail** (model ignores the injection).
4. **Metrics**: Calculates the success rate of injections across the dataset to quantify vulnerability.
#### Additional Methodology:
- **Adversarial Benchmarking**: Implement the Typographic Visual Prompt Injection (TVPI) framework from the referenced paper to systematically evaluate attack success across model size, text factors (size, opacity, position), and target semantics (harmful, biased, neutral).
#### Future Work
- **Defense Mechanisms**: Explore and implement mitigation strategies such as prompt prefixing ("ignore text in image"), input sanitization, adversarial training, and synthetic data augmentation to harden models against visual prompt injection.

## 🌍 Impact
This project highlights a critical and underexplored attack surface in modern AI systems. By open-sourcing the injection methodology and evaluation pipeline, we aim to:
- Raise awareness among developers and researchers about visual prompt injection risks.
- Provide tools for red-teaming and security auditing of vision-language models and AI browsers.
- Catalyze the development of more robust, trustworthy multimodal AI systems that can be safely deployed in high-stakes environments such as healthcare, finance, and autonomous agents.

**Additional Sources:**
- [*ChatGPT's Altas Browser is a Security Nightmare* (youtube.com)](https://www.youtube.com/watch?v=Plzp5z5RsJw)
- [*OpenAI's Atlas browser promises ultimate convenience. But the glossy marketing masks safety risks* (University of Sydney)](https://www.sydney.edu.au/news-opinion/news/2025/10/30/openai-atlas-browser-comes-with-safety-risks.html)
- [*OpenAI warns AI browsers may never be fully secure; says prompt injection may never be solved* (Times of India)](https://timesofindia.indiatimes.com/technology/tech-news/openai-warns-ai-browsers-may-never-be-fully-secure-says-prompt-injection-may-never-be-solved/articleshow/126138136.cms)
- [*Prompt injections as far as the eye can see*](https://simonw.substack.com/p/prompt-injections-as-far-as-the-eye)
- [*THE AI HACK THAT'S BREAKING THE INTERNET: The Prompt Injection Pandemic* (linkedin.com)](https://www.linkedin.com/pulse/ai-hack-thats-breaking-internet-prompt-injection-archie-jackson--0u6sf/)
