# Kısa ara özetler (chunk başına). Hız ve stabilite için madde madde.
CHUNK_PROMPT = """Summarize the following passage in clear academic English as concise bullet points.
- Use 5–8 bullets.
- Preserve key technical terms and important numbers.
- Avoid metadata (journal names, URLs, figure labels, citation markers).
- Only write in English.

Passage:
{text}

Bullet summary:
"""

# Nihai (500+ kelime) akademik özet – SENİN PAYLAŞTIĞIN PROMPT'UN DÜZENLENMİŞ HALİ
FINAL_PROMPT = """You are an expert academic assistant. Using the section-wise summaries of a scientific article provided below, write a comprehensive and well-structured academic summary of the entire paper. Your goal is to integrate and condense the essential information from all sections into a single, cohesive summary that is:

- At least 500 words in length (preferably 550–800 words)
- Written in clear, fluent, formal academic English
- Organized into distinct paragraphs, each addressing one or more of the following aspects:
  • Objective and motivation of the research
  • Methodology and data collection procedures
  • Main results and analytical insights
  • Conclusions, implications, and potential future work
- Avoid inclusion of irrelevant metadata (e.g., journal names, URLs, figure labels such as “Fig-1”, citation markers like “[1]”, or publication headers)
- Do not reproduce the original section titles (e.g., “Introduction”, “Methodology”)
- Focus on conveying the substance of the article’s content without unnecessary repetition
- Ensure a logical and smooth progression between paragraphs through appropriate transitions
- Preserve key technical terms and significant numerical results when relevant
- Only write in English.

Begin your output directly with the summary text. Below are the extracted chunk-wise summaries to be synthesized:

{text}
"""
