Right now, there are several places where "next wire measuring position" is calculated. There should be just one
The idea of chains moving away from known wires is good. Is it worse to generate the list at once?
Maybe it's best to generate the plan for the sequence of the wires and not the coordinates.
Measure auto shouldn't be a case of measure_list. It should start with a known wire (or adjacent to one) and 
move minimizing differences of wire number. 
Measure list should start with the closest wire, then minimize euclidean distances. In this case 
we should come up with a list of wires and positions, but the order doesn't really matter.

Make sure that the refactored code still works.
Fix the servo control so that it runs independently of measuring, 

Three independent threads: wiggling, measuring audio (monitoring amplitude), and plucking (regularly)

Use streaming pesto confidence trigger?