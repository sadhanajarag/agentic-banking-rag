from northstar_rag.llm.rewriter import PROMPT


def test_prompt_takes_question():
    assert PROMPT.input_variables == ["question"]
