def get_dummy_matcher():
    count = 0
    def dymmy_matches(call_depth, chat_id, nl_problem, rephrased_nl_problem, prompt_gen, cal_llm_manager, plotter,
                      parent_func):
        nonlocal count  # Correct usage of nonlocal
        count += 1
        return count < 4
    return dymmy_matches

def formalize_s0(call_depth, nl_problem, cal_llm_manager, verbalizer, prompt_gen, parent_func, plotter):
    node_name = f'formalize_s0({call_depth})'
    if plotter is not None:
        plotter.add_node(node_name, parent_func)

    nl_problem_prompt = prompt_gen.prepare_intro_prompt(nl_problem, verbalizer)
    chat_id = cal_llm_manager.start_new_chat()
    fl_prob = cal_llm_manager.call_llm(chat_id, nl_problem_prompt)
    return process_formal_state_s1(call_depth+1, chat_id, nl_problem, fl_prob, verbalizer, cal_llm_manager, prompt_gen,
                                   node_name, plotter)

def reformalize_after_wrong_syntax_s0s1(call_depth, chat_id, nl_problem, fl_problem, error_message, verbalizer,
                                        cal_llm_manager, prompt_gen, parent_func, plotter):
    node_name = f'reformalize_after_wrong_syntax_s0s1({call_depth})'
    if plotter is not None:
        plotter.add_node(node_name, parent_func)

    problem_prompt_with_error = prompt_gen.prepare_error_prompt(nl_problem, fl_problem, error_message)
    new_yml = cal_llm_manager.call_llm(chat_id, problem_prompt_with_error) # fix this
    return process_formal_state_s1(call_depth+1, chat_id, nl_problem, new_yml, verbalizer, cal_llm_manager, prompt_gen,
                                   node_name, plotter)

def process_formal_state_s1(call_depth, chat_id, nl_problem, fl_problem, verbalizer, cal_llm_manager, prompt_gen,
                            parent_func, plotter):
    node_name = f'process_formal_state_s1({call_depth})'
    if plotter is not None:
        plotter.add_node(node_name, parent_func)

    if verbalizer.is_valid_symbol_valid_syntax(fl_problem):
        nl_of_fl_prob = verbalizer.get_verbalization_of_latest_fl_prob()
        if matches(call_depth + 1, chat_id, nl_problem, nl_of_fl_prob, prompt_gen, cal_llm_manager, plotter,
                   node_name):
            return fl_problem
        else:
            # Simple new try with no history
            return formalize_s0(call_depth + 1, nl_problem, cal_llm_manager, verbalizer, prompt_gen,
                                node_name, plotter)
    elif verbalizer.is_valid_symbol_incorrect_syntax(fl_problem):
        error_msg = verbalizer.get_latest_error_message()
        return reformalize_after_wrong_syntax_s0s1(call_depth + 1, chat_id, nl_problem, fl_problem, error_msg,
                                                   verbalizer, cal_llm_manager, prompt_gen, node_name, plotter)

    elif verbalizer.is_new_symbols_and_valid_syntax(fl_problem):
        for new_symbol in verbalizer.get_new_symbols():
            # new_symbol example: "tangent O A B  # AB is tangent to circle O with radius OA at A"
            # new_sym_inst, nl_description = parse_new_symbol(new_symbol)
            # Start a whole new formalization!
            fl_def_new_sym = formalize_s0(call_depth + 1, new_symbol, cal_llm_manager, verbalizer, prompt_gen,
                                              node_name, plotter)

            if not verbalizer.add_new_symbol(fl_def_new_sym):
                print(f'Warning: Could not add constraint {new_symbol} Check the format of the formal '
                      f'definition\n{fl_def_new_sym}')
            fl_problem = f'{fl_def_new_sym}\n{fl_problem}'

        # TODO: Perhaps you can do a final check if the LLM thinks the final verbalization matches the problem.
        # Else restart/give up
        return fl_problem
    else:
        raise ValueError(f'None of the possibilities satisfied something very wrong with\n{fl_problem}\n'
                         f'Doublecheck if it has fields named "formalization", and "constraints".')

def parse_new_symbol(new_symbol):
    """
    new_symbol: 'tangent O A B  # AB is tangent to circle O with radius OA at A'
    return 'tangent O A B', 'AB is tangent to circle O with radius OA at A'
    """
    inst, description = new_symbol.split('#')
    return inst.strip(), description.strip()

def matches(call_depth, chat_id, nl_problem, rephrased_nl_problem, prompt_gen, cal_llm_manager, plotter, parent_func):
    if plotter is not None:
        plotter.add_node(f'matching_nl({call_depth})', parent_func)

    if cal_llm_manager is None:
        return True
    else:
        match_prompt = prompt_gen.prepare_match_prompt(nl_problem, rephrased_nl_problem)
        if cal_llm_manager.call_llm(chat_id, match_prompt).lower().find('yes') >= 0:
            return True
        else:
            return False


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from prompt_gen.prompt_generator import PromptGenerator
    from verbalize.verbalize_constraints import PreDefConstVerbalizer
    from tool_generation.prompt_gen.chat_manager import ChatHistManager
    from draw_search_tree import DrawSearchTree
    # matches = get_dummy_matcher()
    tree_drawer = DrawSearchTree()
    pre_def_const_file = '../prompt_gen/atomic_constraints.yml'
    verbalizer = PreDefConstVerbalizer(pre_def_const_file)
    prompt_gen = PromptGenerator('../prompt_gen/prob_intro_template.txt', '../prompt_gen/example_problem.yml')
    call_llm_manager = ChatHistManager(None, None, '', '', 0,
                                       True)
    fl_prob = formalize_s0(0,
                           'C is a point on the perpendicular bisector of AB. Prove that AC = BC.',
                           cal_llm_manager=call_llm_manager, verbalizer=verbalizer, prompt_gen=prompt_gen,
                           parent_func='root', plotter=tree_drawer)
    print(fl_prob)
