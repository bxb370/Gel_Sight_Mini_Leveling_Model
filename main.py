"""
• Key rule: if something works, move it to src/
• Key idea: 
	• src/ = reusable building blocks (functions)
	• main.py = experiment runner / pipeline controller
	So instead of “doing everything in main,” you’re really:
orchestrating experiments in main.py using functions from src/
"""

"""
one script that does everything, loads the data, gets the metrics, trains the model, and evaluates the model
"""