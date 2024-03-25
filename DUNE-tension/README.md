# DUNE-tension
automate tension testing for DUNE APAs

Full remote:

1. issue winder command via remote interface (at 127.0.0.1)
    * open connection
    * issue "step" command
    * use selenium
2. actuate servo with pololu micromaestro6
    * https://www.pololu.com/docs/0J40/all
3. detect pitch from laser->photodiode with CREPE
    * pvrecorder to CREPE
4. Log pitch and wire number
    * out to csv, calculate tension as well?

individual wire measurement

1. input wire number, calculate plucking point
2. generate safe G-code to traverse to plucking point
3. send Gcode to interface (how to do this?)


