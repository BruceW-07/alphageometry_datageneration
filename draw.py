# Contributed by github.com/Landau1994
# Modified for issue #42: https://github.com/google-deepmind/alphageometry/issues/42

import unittest
import random
from absl.testing import absltest
import dd
import graph as gh
import problem as pr

# problem_file = "imo_ag_30.txt"
# problem_file = "jgex_ag_231.txt"
problem_file = "new_problem.txt"
problems = pr.Problem.from_txt_file(
    problem_file, 
    to_dict=True, 
    translate=False
)

for key in problems:
    print(key)
    defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
    p = problems[key]
    g, _ = gh.Graph.build_problem(p, defs)
    goal_args = g.names2nodes(p.goal.args)
    gh.nm.draw(
        g.type2nodes[gh.Point],
        g.type2nodes[gh.Line],
        g.type2nodes[gh.Circle],
        g.type2nodes[gh.Segment],
        #block=True,
        save_to="./output/plot/"+problem_file+"/"+key+".jpg")