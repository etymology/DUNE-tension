from datetime import datetime
import os
import csv
import random
import time

def log_frequency_data(data, filename):
    """
    Log data into a CSV file.
    
    :param data: Dictionary containing field-value pairs to log.
    :param filename: Name of the file to log the data into.
    """
    # If filename does not contain a path, use the current working directory
    if not os.path.dirname(filename):
        directory = os.getcwd()  # Get current directory
    else:
        directory = os.path.dirname(filename)
    
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Full path for the file
    full_path = os.path.join(directory, os.path.basename(filename))

    # Ensure the 'time' field is included in the data
    data['time'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Open file safely with handling exceptions
    try:
        with open(full_path, 'a', newline='') as csvfile:
            fieldnames = list(data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # If file is empty, write header
            if os.stat(full_path).st_size == 0:
                writer.writeheader()
            writer.writerow(data)
            # print("Data logged successfully.")
    except IOError as e:
        print(f"Error opening or writing to file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def load_csv_to_dict(file_path):
    data = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        first_column = reader.fieldnames[0]
        for row in reader:
            key = int(row.pop(first_column))
            data[key] = {k: float(v) for k, v in row.items()}
    return data

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute.")
        return result  # Return the result of the function
    return wrapper

def gaussian_wiggle(wire_y, wiggle_step=.2):
    while True:
        yield random.gauss(wire_y, wiggle_step)

def stepwise_wiggle(wire_y, wiggle_step = 0.1):
    i = 0
    while True:
        yield wire_y + (-1)**(i // 2 + 1) * (i // 2 + 1) * wiggle_step
        i += 1

def get_wiggle_generator(wiggle_type, wire_y, wiggle_step=0.1):
    if wiggle_type == 'gaussian':
        return gaussian_wiggle(wire_y)
    else:
        return stepwise_wiggle(wire_y)


if __name__ == "__main__":
    # Test the wiggle generator
    wire_y = 0
    wg = get_wiggle_generator('gaussian', wire_y)
    for _ in range(10):
        print(next(wg))

