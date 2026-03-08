from __future__ import annotations

import asyncio
import os
import re
from collections import deque
from collections import OrderedDict
from dataclasses import dataclass

from groq import Groq


SYSTEM_PROMPT = (
    "You are an ASL (American Sign Language) interpreter. "
    "You receive tokens that are either ASL sign labels (e.g. OPEN_PALM, THUMB_UP, ILY) "
    "or individual fingerspelled letters (e.g. H E L L O → Hello, W A T E R → Water). "
    "Tokens wrapped in [SPELL:...] are pre-grouped fingerspelled words — decode them directly "
    "(e.g. [SPELL:WATER] → Water, [SPELL:HELLO] → Hello, [SPELL:NAME] → Name). "
    "Convert the full token sequence into ONE short, natural English sentence. "
    "Use conversation history for context only — do not repeat previous sentences. "
    "Rules you must follow without exception:\n"
    "- Output ONLY the sentence. Nothing else.\n"
    "- No explanations, no reasoning, no preamble, no meta-commentary.\n"
    "- No phrases like 'I interpret', 'the sequence means', 'with token', 'as:'.\n"
    "- No quotation marks around the sentence.\n"
    "- If you cannot form a sentence, output a single best-guess word.\n"
    "Your entire response must be the sentence itself and nothing more."
)

# Language-specific addenda appended to the system prompt when a non-English
# output language is selected.  Keyed by ISO-639-1 code.
_LANGUAGE_ADDENDUM: dict[str, str] = {
    "hi": (
        "IMPORTANT: Output the sentence in Hindi (Devanagari script). "
        "Do NOT output English. The entire response must be a natural Hindi sentence. "
        "Example: if the signs mean 'thank you', output 'धन्यवाद'."
    ),
}

# Scenario-specific addenda that prime the LLM with the right context so it
# picks the most natural phrasing for the situation.  Keyed by scenario name.
_SCENARIO_ADDENDUM: dict[str, str] = {
    "interview": (
        "CONTEXT: This is a job interview. "
        "The signer is a candidate communicating with an interviewer about their experience, skills, "
        "and suitability for a role. Produce professional, formal sentences a candidate would say. "
        "Examples: 'I have five years of experience in software development.', "
        "'My greatest strength is problem-solving under pressure.', "
        "'I am very interested in this position.', "
        "'When can I expect to hear back from you?'"
    ),
    "banking": (
        "CONTEXT: This is a banking or financial services setting. "
        "The signer is a customer at a bank interacting with a teller or advisor. "
        "Produce sentences about transactions, account inquiries, or financial requests. "
        "Examples: 'I would like to withdraw two hundred dollars.', "
        "'Please check my account balance.', "
        "'I need to transfer funds to another account.', "
        "'Can I open a new savings account?', "
        "'I would like to deposit this cheque.'"
    ),
    "medical": (
        "CONTEXT: This is a medical or healthcare setting such as a doctor's office, hospital, or pharmacy. "
        "The signer is a patient communicating with healthcare staff about symptoms, medications, or appointments. "
        "Examples: 'I have been having headaches for three days.', "
        "'I need a refill on my prescription.', "
        "'My appointment is at two in the afternoon.', "
        "'I am allergic to penicillin.', "
        "'The pain is on my right side.'"
    ),
    "restaurant": (
        "CONTEXT: This is a restaurant or food ordering setting. "
        "The signer is a customer communicating with waitstaff or at an ordering counter. "
        "Produce sentences about ordering food, dietary preferences, or dining. "
        "Examples: 'I would like the grilled chicken, please.', "
        "'Can I see the menu?', "
        "'No onions, please, I am allergic.', "
        "'The check, please.', "
        "'Table for two.'"
    ),
}


@dataclass(slots=True)
class GroqConfig:
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    max_tokens: int = 40
    top_p: float = 0.9
    cache_size: int = 1024
    context_turns: int = 6


class ConversationContext:
    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max(1, max_turns)
        self._turns: deque[tuple[str, str]] = deque(maxlen=self.max_turns)

    def add_turn(self, tokens: list[str], sentence: str) -> None:
        token_text = " ".join(tokens).strip()
        sentence_text = sentence.strip()
        if not token_text or not sentence_text:
            return
        self._turns.append((token_text, sentence_text))

    def reset(self) -> None:
        self._turns.clear()

    def turns(self) -> list[tuple[str, str]]:
        return list(self._turns)


def _preprocess_tokens(tokens: list[str]) -> list[str]:
    """
    Detect runs of single-letter tokens (fingerspelled letters) and merge
    them into annotated words so the LLM can decode them more reliably.

    Example: ["OPEN_PALM", "W", "A", "T", "E", "R"] → ["OPEN_PALM", "[SPELL:WATER]"]
    Single isolated letters are left unchanged.
    """
    result: list[str] = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            run: list[str] = []
            while i < len(tokens) and len(tokens[i]) == 1 and tokens[i].isalpha():
                run.append(tokens[i])
                i += 1
            if len(run) >= 2:
                result.append(f"[SPELL:{''.join(run)}]")
            else:
                result.extend(run)
        else:
            result.append(tokens[i])
            i += 1
    return result


class LLMService:
    def __init__(self, config: GroqConfig | None = None) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY environment variable.")
        self.config = config or GroqConfig()
        context_turns = os.getenv("CONVERSATION_CONTEXT_TURNS")
        if context_turns is not None:
            self.config.context_turns = max(1, int(context_turns))
        self.client = Groq(api_key=api_key)
        self.cache: OrderedDict[tuple[str, ...], str] = OrderedDict()

    async def tokens_to_sentence(self, tokens: list[str]) -> str:
        return await self.tokens_to_sentence_with_context(tokens=tokens, context=None)

    async def tokens_to_sentence_with_context(
        self,
        tokens: list[str],
        context: ConversationContext | None,
    ) -> str:
        key = self._cache_key(tokens=tokens, context=context)
        cached = self.cache.get(key)
        if cached:
            self.cache.move_to_end(key)
            return cached

        sentence = await asyncio.to_thread(self._generate, tokens, context)
        if context is not None:
            context.add_turn(tokens=tokens, sentence=sentence)
        self._cache_set(key, sentence)
        return sentence

    @staticmethod
    def _cache_key(
        tokens: list[str],
        context: ConversationContext | None,
        scenario: str = "general",
        custom_scenario: str = "",
    ) -> tuple[str, ...]:
        key: list[str] = ["TOKENS", *tokens]
        if scenario and scenario != "general":
            key += ["SCENARIO", scenario]
            if scenario == "custom" and custom_scenario:
                key += ["CUSTOM", custom_scenario[:100]]
        if context is not None:
            for token_text, sentence in context.turns():
                key += ["CTX", token_text, sentence]
        return tuple(key)

    def _cache_set(self, key: tuple[str, ...], value: str) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.config.cache_size:
            self.cache.popitem(last=False)

    def _build_messages(
        self,
        tokens: list[str],
        context: ConversationContext | None,
        language: str = "en",
        scenario: str = "general",
        custom_scenario: str = "",
    ) -> list[dict]:
        """Construct the chat message list for a token sequence."""
        sys_content = SYSTEM_PROMPT

        # Inject scenario context so the LLM picks situationally appropriate phrasing
        if scenario == "custom" and custom_scenario.strip():
            scenario_addendum = (
                f"CONTEXT: This conversation is happening in the following setting: "
                f"{custom_scenario.strip()}. "
                "Produce sentences that are natural and appropriate for this context."
            )
            sys_content = sys_content + "\n" + scenario_addendum
        elif scenario in _SCENARIO_ADDENDUM:
            sys_content = sys_content + "\n" + _SCENARIO_ADDENDUM[scenario]

        # Language override comes last (highest priority)
        lang_addendum = _LANGUAGE_ADDENDUM.get(language, "")
        if lang_addendum:
            sys_content = sys_content + "\n" + lang_addendum

        messages: list[dict] = [{"role": "system", "content": sys_content}]
        if context is not None:
            for token_text, sentence in context.turns():
                messages.append({"role": "user", "content": f"Tokens: {token_text}"})
                messages.append({"role": "assistant", "content": sentence})
        processed = _preprocess_tokens(tokens)
        messages.append({"role": "user", "content": f"Tokens: {' '.join(processed)}"})
        return messages

    def _generate(
        self,
        tokens: list[str],
        context: ConversationContext | None,
        language: str = "en",
        scenario: str = "general",
        custom_scenario: str = "",
    ) -> str:
        """Build a multi-turn chat and call Groq. Past turns become real
        user/assistant message pairs so the model sees natural conversation flow."""
        messages = self._build_messages(tokens, context, language=language, scenario=scenario, custom_scenario=custom_scenario)
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return _clean_llm_output(raw)

    def stream_tokens(
        self,
        tokens: list[str],
        context: ConversationContext | None,
        language: str = "en",
        scenario: str = "general",
        custom_scenario: str = "",
    ):
        """Streaming generator — yields raw text chunks as they arrive from Groq.
        Call _clean_llm_output on the accumulated result at the end."""
        messages = self._build_messages(tokens, context, language=language, scenario=scenario, custom_scenario=custom_scenario)
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stream=True,
        )
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta


_META_PATTERN = re.compile(
    r".+?(?:"
    r"interpret(?:ed|s)?(?: the sequence)? as[:\s]+"
    r"|(?:the )?sequence (?:means?|is)[:\s]+"
    r"|translates? to[:\s]+"
    r"|(?:can be )?(?:read|expressed) as[:\s]+"
    r"|with (?:the )?(?:additional )?token .+?[:,]\s*"
    r"|output[:\s]+"
    r"|sentence[:\s]+"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def _clean_llm_output(text: str) -> str:
    """
    Strip meta-reasoning the model sometimes leaks before the actual sentence.

    Handles patterns like:
      'With the additional token THUMB_UP, I can interpret the sequence as: "I know, yes."'
      'The sequence means: Hello there.'
      'I interpret this as: Good morning.'
    """
    # If the output contains a known meta-prefix, strip everything up to and
    # including the prefix, then take what follows.
    cleaned = _META_PATTERN.sub("", text).strip()
    if cleaned:
        # Remove any surrounding quotes left behind
        cleaned = cleaned.strip("\"'").strip()
        return cleaned

    # Fallback: if pattern matched nothing meaningful, return original
    return text
