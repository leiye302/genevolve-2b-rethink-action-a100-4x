import json
import hashlib
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from genevolve_repro.action_protocol import (
    ACTION_PROTOCOL, FORMAT_ERROR, FORMAT_ERROR_NATIVE, commands, command_to_native,
    convert_instruction, convert_messages, native_to_command, parse_generate, parse_tool,
)
from genevolve_repro.action_agent import CommandAgent
from genevolve_repro.common import ROOT, read_json
from genevolve_repro.policy import action_indices
from genevolve.agent import GenEvolveAgent, _parse_answer, _parse_tool_call
from genevolve.system_prompt import SYSTEM_PROMPT, FINAL_STEP_MESSAGE
from test_actor_text import CharacterTokenizer


class ActionProtocolTest(unittest.TestCase):
    def test_all_commands_keep_argument_text_and_current_reasoning(self):
        prefix = '<think>Use only available evidence.</think>\n<RETHINK>'
        for name, args in [
            ('search', '{"queries": ["cat", "dog"], "top_k": 3}'),
            ('image_search', '{"query": "a quoted \\" word", "top_k": 5}'),
            ('query_knowledge', '{"skill_name":"spatial_layout"}'),
            ('generate', '{"gen_prompt":"Draw a cat", "reference_images":[{"img_id":"IMG_001"}]}'),
        ]:
            # JSON escaping in a literal fixture is independently validated.
            if name == 'image_search':
                args = json.dumps({'query': 'a quoted " word', 'top_k': 5})
            native = ('<answer>' + args + '</answer>' if name == 'generate' else
                      '<tool_call>{"name":' + json.dumps(name) + ',"arguments":' + args + '}</tool_call>')
            result = native_to_command(prefix + native)
            self.assertEqual(result, prefix + '<action>' + name + ' ' + args + '</action>')
            command, = commands(result)
            self.assertEqual(command.arguments, json.loads(args))
            inverse = command_to_native(result)
            self.assertEqual(_parse_answer(inverse), _parse_answer(native))
            self.assertEqual(_parse_tool_call(inverse), _parse_tool_call(native))

    def test_quoted_tags_nested_json_and_unicode(self):
        args = {'gen_prompt': 'literal </action> and <action>search {} 中文',
                'reference_images': [{'img_id': 'IMG_001', 'note': 'a { } " quote'}]}
        text = '<action>generate ' + json.dumps(args, ensure_ascii=False) + '</action>'
        self.assertEqual(parse_generate(text), args)
        self.assertIsNone(parse_tool(text))
        self.assertEqual(len(list(commands(text))), 1)
        tok = CharacterTokenizer()
        # Byte-style fixtures support Latin-1; use ASCII JSON escapes here.
        ascii_text = '<action>generate ' + json.dumps(args) + '</action>'
        ids = tok('<think>Keep this separate.</think><RETHINK>' + ascii_text + '<eos>')['input_ids']
        self.assertEqual(tok.decode([ids[i] for i in action_indices(tok, ids)]), ascii_text)

    def test_no_new_reasoning_gate_or_automatic_native_conversion_online(self):
        self.assertEqual(parse_tool('<action>search {"queries":["a"]}</action>'), ('search', {'queries':['a']}))
        self.assertIsNone(parse_tool('<tool_call>{"name":"search","arguments":{}}</tool_call>'))
        self.assertIsNone(parse_generate('<answer>{"gen_prompt":"a"}</answer>'))
        self.assertIsNone(parse_tool('<action>{"name":"broken":0}</action>'))
        self.assertIsNone(parse_tool('<action>search {"queries":}</action>'))
        self.assertEqual(parse_tool('<action>unknown {}</action>'), ('unknown', {}))

    def test_only_protocol_instruction_changes_and_final_override_preserved(self):
        changed = convert_instruction(SYSTEM_PROMPT)
        self.assertIn('<action><function-name> <args-json-object></action>', changed)
        self.assertNotIn('<tool_call>', changed)
        self.assertNotIn('<answer>', changed)
        self.assertEqual(changed[changed.index('<tools>'):changed.index('</tools>')],
                         SYSTEM_PROMPT[SYSTEM_PROMPT.index('<tools>'):SYSTEM_PROMPT.index('</tools>')])
        last = convert_instruction(FINAL_STEP_MESSAGE)
        self.assertIn('Do NOT write <think>', last)
        self.assertIn('ABSOLUTELY FORBIDDEN', last)
        self.assertIn('<action>generate ', last)

    def test_history_preserves_errors_observations_and_executed_in_reason_action(self):
        inner = '<tool_call>{"name":"search","arguments":{"queries":["first"]}}</tool_call>'
        outer = '<tool_call>{"name":"search","arguments":{"queries":["second"]}}</tool_call>'
        broken = '<think>x</think><tool_call>{"name":"broken":0}</tool_call>'
        rows = [
            {'role':'system','content':SYSTEM_PROMPT},
            {'role':'user','content':'Quoted user text <answer> stays unchanged'},
            {'role':'assistant','content':'<think>'+inner+'</think>'+outer},
            {'role':'user','content':'<tool_response>first search observation</tool_response>'},
            {'role':'assistant','content':broken},
            {'role':'user','content':FORMAT_ERROR_NATIVE},
            {'role':'user','content':FINAL_STEP_MESSAGE},
        ]
        original = deepcopy(rows)
        out = convert_messages(rows)
        self.assertEqual(rows, original)
        self.assertEqual(out[1], rows[1])
        self.assertEqual(out[3], rows[3])
        self.assertEqual(parse_tool(out[2]['content']), _parse_tool_call(rows[2]['content']))
        self.assertIsNone(parse_tool(out[4]['content']))
        self.assertIn('{"name":"broken":0}', out[4]['content'])
        self.assertEqual(out[5]['content'], FORMAT_ERROR)
        with self.assertRaises(ValueError): native_to_command(broken)

    def paired_loop(self, outputs, max_rounds=11, fail_search=False):
        results, calls = [], []
        for cls in (GenEvolveAgent, CommandAgent):
            search = Mock(side_effect=RuntimeError('fixture failure')) if fail_search else Mock(return_value='evidence')
            images = Mock(return_value=[{'title':'bird', 'local_path':f'/fixture/ref{i}.png'} for i in range(3)])
            knowledge = Mock(return_value='spatial layout guidance')
            agent = cls(max_rounds=max_rounds, text_search_tool=SimpleNamespace(call=search),
                        image_search_tool=SimpleNamespace(call=images), knowledge_tool=SimpleNamespace(call=knowledge))
            self.addCleanup(agent._client.close)
            answers = iter(outputs)
            def respond(_messages):
                item = next(answers)
                if isinstance(item, Exception): raise item
                return (native_to_command(item, preserve_invalid=True, convert_in_reasoning=True)
                        if cls is CommandAgent else item)
            agent._call_llm = respond
            results.append(agent.run('Draw a bird.'))
            calls.append([search.call_args_list, images.call_args_list, knowledge.call_args_list])
        old, new = results
        self.assertEqual(convert_messages(old.messages), new.messages)
        self.assertEqual({k:v for k,v in old.to_dict().items() if k!='messages'},
                         {k:v for k,v in new.to_dict().items() if k!='messages'})
        self.assertEqual(calls[0], calls[1])
        return new

    def test_same_loop_all_tools_reference_selection_and_termination(self):
        result = self.paired_loop([
            '<think>x</think><tool_call>{"name":"search","arguments":{"queries":["bird"],"top_k":2}}</tool_call>',
            '<think>x</think><RETHINK><tool_call>{"name":"image_search","arguments":{"query":"bird"}}</tool_call>',
            '<think>x</think><tool_call>{"name":"query_knowledge","arguments":{"skill_name":"spatial_layout"}}</tool_call>',
            '<think>x</think><answer>{"gen_prompt":"Use the first reference image.","reference_images":[{"img_id":"IMG_003"},{"img_id":"IMG_002"},{"img_id":"IMG_001"}]}</answer>',
        ])
        self.assertEqual(result.termination, 'answer')
        self.assertEqual([r['img_id'] for r in result.reference_images], ['IMG_001','IMG_002'])

    def test_same_format_error_last_round_and_tool_failure_behaviors(self):
        self.paired_loop(['invalid', '<answer>{"gen_prompt":"draw"}</answer>'], max_rounds=2)
        result = self.paired_loop(['invalid', 'invalid'], max_rounds=2)
        self.assertEqual(result.termination, 'no_answer')
        result = self.paired_loop([
            '<tool_call>{"name":"search","arguments":{}}</tool_call>',
            '<answer>{"gen_prompt":"draw"}</answer>',
        ], fail_search=True)
        self.assertIn('[tool error] search', result.messages[3]['content'])
        self.paired_loop([RuntimeError('fixture LLM error')])

    def test_action_mask_includes_command_and_payload_not_reasoning_or_marker(self):
        tok = CharacterTokenizer()
        for name in ('search', 'image_search', 'query_knowledge', 'generate'):
            action = '<action>'+name+' {}</action>'
            text = '<think>Reason first.</think>\n<RETHINK>'+action+'<eos>'
            ids = tok(text)['input_ids']
            selected = action_indices(tok, ids)
            self.assertEqual(tok.decode([ids[i] for i in selected]), action)
            self.assertNotIn(tok.specials['<RETHINK>'], [ids[i] for i in selected])

    def test_no_type_bce_or_new_training_settings(self):
        cfg = read_json(ROOT/'configs/route_sft.json')
        self.assertEqual(cfg['action_protocol'], ACTION_PROTOCOL)
        for key, value in {'parent_step':550,'world_size':4,'micro_batch':1,'good_per_batch':6,
                           'bad_per_batch':2,'batches_per_epoch':297,'epochs':2,'max_length':12288,
                           'loss_reduction':'sample_mean','suffix_ce_weight':1.0,'route_lambda':0.1,
                           'reference_kl_coef':0.0,'lr':2e-6,'seed':0}.items():
            self.assertEqual(cfg[key], value, key)
        self.assertNotIn('type_lambda', cfg)

    def test_frozen_core_files_and_release_identity(self):
        report = read_json(ROOT/'data/method_diff.json')
        for name, expected in report['unchanged_core_lf_sha256'].items():
            text = (ROOT/name).read_text(encoding='utf-8').replace('\r\n','\n').replace('\r','\n')
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), expected, name)
        audit = read_json(ROOT/'data/protocol_audit.json')
        cfg = read_json(ROOT/'configs/route_sft.json')
        lock = read_json(ROOT/'configs/route_release.json')
        self.assertEqual(cfg['manifest_sha256'], lock['files']['manifest.json']['sha256'])
        self.assertEqual(cfg['manifest_sha256'], audit['converted_manifest_sha256'])
        self.assertEqual((audit['rows'], audit['target_semantics_checked'], audit['history_actions_semantics_checked']),
                         (2376, 2376, 6091))


if __name__ == '__main__':
    unittest.main()
