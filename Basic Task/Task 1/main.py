import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

tasks = pd.DataFrame(columns=['description', 'priority'])

try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)
model.fit(tasks['description'], tasks['priority'])

def add_task(description, priority):
    global tasks  # Declare tasks as a global variable
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()


# Function to remove a task by description
def remove_task(description):
    tasks = tasks[tasks['description'] != description]
    save_tasks()

# Function to list all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)

def recommend_task():
    if tasks.empty:  # Ensure 'tasks' is not empty before proceeding
        print("No tasks available for recommendations.")
        return

    # Filter high-priority tasks
    high_priority_tasks = tasks[tasks['priority'] == 'High']

    if high_priority_tasks.empty:  # Check if high-priority tasks exist
        print("No high-priority tasks available for recommendation.")
        return

    # Convert the 'description' column to a list before using random.choice
    task_list = high_priority_tasks['description'].tolist()

    if not task_list:  # Ensure there's at least one task in the list
        print("No valid descriptions found in high-priority tasks.")
        return

    # Choose a random task from the filtered list
    random_task = random.choice(task_list)
    
    print(f"Recommended task: {random_task} - Priority: High")

# Main menu
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        add_task(description, priority)
        print("Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)
        print("Task removed successfully.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")
