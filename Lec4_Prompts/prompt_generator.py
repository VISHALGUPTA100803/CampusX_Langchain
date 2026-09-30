from langchain_core.prompts import PromptTemplate
import json

template = PromptTemplate(
    template = """
 Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  

If you are not familiar enough with this specific paper to answer accurately, say so honestly rather than fabricating details.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=["paper_input","style_input", "length_input" ],
validate_template=True
)

with open("template.json", "w", encoding="utf-8") as file:
    json.dump(template.to_json(), file, indent=2)

# If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

# What this tells us

# The fix worked with the mathematical-details section completely unchanged in the prompt. So the equations/code-snippet instruction itself was never actually the problem. My theory that "Technical/Mathematical/Code-Oriented specifically trip on the precision demand" was wrong, or at least incomplete.

# What actually seems to matter: the framing of the escape hatch itself, not which section it's attached to.

# Compare the two versions of the refusal condition:

# Original (fails on 3/4 styles):

# If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

# Your one-line fix (works on all 4):

# If you are not familiar enough with this specific paper to answer accurately, say so honestly rather than fabricating details.

# These sound similar but frame the decision completely differently:

# "Information not available in the paper" — frames this as a fact-checking task against an external source. Since no source was ever given, the honest, literal answer to "is X available in the paper" is always technically "I don't have the paper to check," making refusal the safe, correct reading regardless of style. This framing invites the model to treat itself as a retrieval system with nothing to retrieve from.
# "Not familiar enough... to answer accurately" — frames this as a self-assessment of the model's own knowledge, not a lookup against a document. This is a question the model can actually answer meaningfully: "do I know enough about Attention Is All You Need to speak confidently?" For a paper this famous, the honest self-assessment is "yes, I know this well," so it proceeds.
# Why this explains the style-dependent pattern from before

# My original theory said "precision-demanding styles trigger refusal." But actually, I think what was really happening: the word "available" combined with "in the paper" primed the model into document-verification mode across the board, and Beginner-Friendly styles just happened to produce answers general enough that this framing didn't get triggered as hard (less specific claims = less to "verify"), while Technical/Math/Code produced more specific, checkable-sounding claims, which more strongly activated the "wait, can I actually verify this specific detail against the source?" hesitation. Same underlying cause, just manifesting more visibly in styles that demand more specificity.

# Your fix removes the "verify against a document" framing entirely, replacing it with "assess your own confidence", so it doesn't matter how specific or precise the requested style is anymore, there's no document to check against in the model's mental model of the task at all now.

# Revised, more accurate conclusion

# The critical lever was never the math/code section or the style variable. It was the wording of the refusal condition itself. "Not available in the source" and "not familiar enough" sound almost synonymous to a human reader, but they point the model toward two entirely different mental tasks, document-verification versus self-knowledge-assessment, and only one of those tasks is actually answerable given what's in your prompt. This is a genuinely subtle and useful prompt-engineering finding: precise wording of an escape-hatch/refusal clause can matter more than which section of the prompt it's near, or how demanding the requested output style is.