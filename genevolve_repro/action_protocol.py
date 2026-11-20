"""Lossless command-first serialization; no new decisions, tools or token IDs."""

from dataclasses import dataclass
import json
import re


ACTION_PROTOCOL = "command-first-action-v1"
NATIVE_BLOCK = re.compile(r"<(?P<tag>tool_call|answer)>\s*(?P<body>\{.*?\})\s*</(?P=tag)>", re.S)
COMMAND_OPEN = re.compile(r"<action>\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+")
DECODER = json.JSONDecoder()
FORMAT_ERROR_NATIVE = ("<tool_response>\nFormat error: must output one "
                       "<tool_call>{...}</tool_call> or <answer>{...}</answer>.\n</tool_response>")
FORMAT_ERROR = ("<tool_response>\nFormat error: must output one "
                "<action>TOOL_NAME {...}</action> or <action>generate {...}</action>.\n</tool_response>")


@dataclass(frozen=True)
class Command:
    name: str
    arguments: dict
    start: int
    end: int
    payload: str


def commands(text):
    """Parse JSON before finding the closing tag, including tags inside strings.

    Like the pinned native parser, a complete action can occur after reasoning.
    This is not a new grammar-constrained decoder or a reasoning-validity gate.
    """
    end = 0
    for match in COMMAND_OPEN.finditer(text):
        if match.start() < end:
            continue
        try:
            value, stop = DECODER.raw_decode(text, match.end())
        except ValueError:
            continue
        close = stop
        while close < len(text) and text[close].isspace():
            close += 1
        if not isinstance(value, dict) or not text.startswith("</action>", close):
            continue
        end = close + len("</action>")
        yield Command(match["name"], value, match.start(), end, text[match.end():stop])


def parse_generate(text):
    # Retain the native final-program priority if a malformed multi-action
    # response happens to contain both a tool and a final program.
    return next((c.arguments for c in commands(text) if c.name == "generate"), None)


def parse_tool(text):
    return next(((c.name, c.arguments) for c in commands(text) if c.name != "generate"), None)


def _member_source(body, wanted):
    """Return an object's value verbatim, not a normalized JSON reserialization."""
    i = body.index("{") + 1
    found = []
    while True:
        while i < len(body) and (body[i].isspace() or body[i] == ","):
            i += 1
        if body[i] == "}":
            break
        key, i = DECODER.raw_decode(body, i)
        while body[i].isspace():
            i += 1
        if body[i] != ":":
            raise ValueError("Malformed native object member")
        i += 1
        while body[i].isspace():
            i += 1
        begin = i
        _, i = DECODER.raw_decode(body, i)
        if key == wanted:
            found.append(body[begin:i])
    if len(found) != 1:
        raise ValueError(f"Expected exactly one native {wanted} member")
    return found[0]


def native_to_command(text, *, preserve_invalid=False, convert_in_reasoning=False):
    """Convert real native action blocks; preserve every other character."""
    def convert(match):
        body = match["body"]
        try:
            value = json.loads(body)
        except ValueError:
            if not preserve_invalid:
                raise
            # Historical syntax failures stay failures. Do not guess a name,
            # fix JSON or fabricate a tool result. This deliberately has no
            # command name, so both the native and new parser reject it.
            return f"<action>{body}</action>"
        if not isinstance(value, dict):
            raise ValueError("Native action is not a JSON object")
        if match["tag"] == "answer":
            name, arguments = "generate", body
        else:
            name = value.get("name")
            if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                    or name == "generate" or set(value) != {"name", "arguments"}
                    or not isinstance(value["arguments"], dict)):
                raise ValueError("Unsupported native tool envelope; do not silently rewrite it")
            arguments = _member_source(body, "arguments")
        return f"<action>{name} {arguments}</action>"

    boundary = -1 if convert_in_reasoning else text.find("</think>")
    boundary = boundary + len("</think>") if boundary >= 0 else 0
    return text[:boundary] + NATIVE_BLOCK.sub(convert, text[boundary:])


def command_to_native(text):
    """Semantic inverse for auditing/comparison only, never used to generate."""
    for command in reversed(list(commands(text))):
        if command.name == "generate":
            block = f"<answer>{command.payload}</answer>"
        else:
            block = ("<tool_call>{\"name\":" + json.dumps(command.name)
                     + ",\"arguments\":" + command.payload + "}</tool_call>")
        text = text[:command.start] + block + text[command.end:]
    return text


def convert_instruction(text):
    """Change only native format references, not search/termination instructions."""
    text = text.replace(
        "For each function call, return JSON within <tool_call></tool_call> XML tags:",
        "For each function call, return its name followed by its arguments JSON within <action></action> XML tags:",
    )
    text = text.replace(
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>',
        '<action><function-name> <args-json-object></action>',
    )
    return (text.replace("<tool_call>", "<action>TOOL_NAME ")
            .replace("</tool_call>", "</action>")
            .replace("<answer>", "<action>generate ")
            .replace("</answer>", "</action>"))


def convert_messages(messages):
    """Offline same-history conversion, shared with the online instructions.

    Observations and user requests are not rewritten. Only a system message,
    assistant action blocks and the exact native format/final-step controls
    are protocol-bearing. No future feedback, images or hints are inserted.
    """
    from .common import load_upstream
    load_upstream()
    from genevolve.system_prompt import FINAL_STEP_MESSAGE

    converted = []
    for message in messages:
        item = dict(message)
        text = item["content"]
        if item["role"] == "system":
            text = convert_instruction(text)
        elif item["role"] == "assistant":
            # The native executor searches the whole response, including any
            # literal action examples inside historical reasoning. Convert
            # those protocol examples too, preserving the executed action and
            # its actual observation. Current conditioned reasoning is not
            # part of messages and remains byte/token-identical in the pack.
            text = native_to_command(text, preserve_invalid=True, convert_in_reasoning=True)
        elif text == FINAL_STEP_MESSAGE:
            text = convert_instruction(text)
        elif text == FORMAT_ERROR_NATIVE:
            text = FORMAT_ERROR
        item["content"] = text
        converted.append(item)
    return converted
