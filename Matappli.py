import pulp
import customtkinter as ctk

# Function to solve the optimization problem
def solve_optimization():
    try:
        # Get values from user input
        profit_a = float(entry_profit_a.get())
        profit_b = float(entry_profit_b.get())
        profit_c = float(entry_profit_c.get())
        storage_capacity = float(entry_storage_capacity.get())
        max_a = int(entry_max_a.get())
        max_b = int(entry_max_b.get())
        max_c = int(entry_max_c.get())
        min_a = int(entry_min_a.get())
        min_b = int(entry_min_b.get())
        min_c = int(entry_min_c.get())
    except ValueError:
        label_result.configure(text="Please enter valid numeric values.")
        return

    # Define the Linear Programming problem
    prob = pulp.LpProblem("Inventory_Optimization", pulp.LpMaximize)

    # Define decision variables (amount of each product to order)
    A = pulp.LpVariable("A", lowBound=min_a, upBound=max_a, cat='Integer')  # Product A
    B = pulp.LpVariable("B", lowBound=min_b, upBound=max_b, cat='Integer')  # Product B
    C = pulp.LpVariable("C", lowBound=min_c, upBound=max_c, cat='Integer')  # Product C

    # Objective function (maximize profit)
    prob += profit_a * A + profit_b * B + profit_c * C, "Total Profit"

    # Constraints
    prob += 5 * A + 3 * B + 2 * C <= storage_capacity, "Storage Capacity"
    prob += A >= min_a, "Minimum Demand for A"
    prob += B >= min_b, "Minimum Demand for B"
    prob += C >= min_c, "Minimum Demand for C"

    # Solve the problem
    prob.solve()

    # Check the result status
    if pulp.LpStatus[prob.status] == "Optimal":
        result_text = f"Optimal solution:\n"
        result_text += f"A: {A.varValue} units\n"
        result_text += f"B: {B.varValue} units\n"
        result_text += f"C: {C.varValue} units\n"
        result_text += f"Total Profit: ${pulp.value(prob.objective):.2f}"
        label_result.configure(text=result_text)
    else:
        label_result.configure(text="No optimal solution found.")

# Create the main window
root = ctk.CTk()
root.title("Inventory Management Optimization")
root.geometry("500x600")

# Set the appearance mode
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Create and place widgets in the window
label_profit_a = ctk.CTkLabel(root, text="Profit per unit of Product A ($):")
label_profit_a.pack(pady=5)
entry_profit_a = ctk.CTkEntry(root)
entry_profit_a.pack(pady=5)

label_profit_b = ctk.CTkLabel(root, text="Profit per unit of Product B ($):")
label_profit_b.pack(pady=5)
entry_profit_b = ctk.CTkEntry(root)
entry_profit_b.pack(pady=5)

label_profit_c = ctk.CTkLabel(root, text="Profit per unit of Product C ($):")
label_profit_c.pack(pady=5)
entry_profit_c = ctk.CTkEntry(root)
entry_profit_c.pack(pady=5)

label_storage_capacity = ctk.CTkLabel(root, text="Storage Capacity (cubic meters):")
label_storage_capacity.pack(pady=5)
entry_storage_capacity = ctk.CTkEntry(root)
entry_storage_capacity.pack(pady=5)

label_max_a = ctk.CTkLabel(root, text="Max units of Product A:")
label_max_a.pack(pady=5)
entry_max_a = ctk.CTkEntry(root)
entry_max_a.pack(pady=5)

label_max_b = ctk.CTkLabel(root, text="Max units of Product B:")
label_max_b.pack(pady=5)
entry_max_b = ctk.CTkEntry(root)
entry_max_b.pack(pady=5)

label_max_c = ctk.CTkLabel(root, text="Max units of Product C:")
label_max_c.pack(pady=5)
entry_max_c = ctk.CTkEntry(root)
entry_max_c.pack(pady=5)

label_min_a = ctk.CTkLabel(root, text="Min units of Product A:")
label_min_a.pack(pady=5)
entry_min_a = ctk.CTkEntry(root)
entry_min_a.pack(pady=5)

label_min_b = ctk.CTkLabel(root, text="Min units of Product B:")
label_min_b.pack(pady=5)
entry_min_b = ctk.CTkEntry(root)
entry_min_b.pack(pady=5)

label_min_c = ctk.CTkLabel(root, text="Min units of Product C:")
label_min_c.pack(pady=5)
entry_min_c = ctk.CTkEntry(root)
entry_min_c.pack(pady=5)

# Button to solve the problem
button_solve = ctk.CTkButton(root, text="Solve Optimization", command=solve_optimization)
button_solve.pack(pady=20)

# Label to show the result
label_result = ctk.CTkLabel(root, text="Results will be displayed here", width=300, height=50)
label_result.pack(pady=10)

# Run the application
root.mainloop()
