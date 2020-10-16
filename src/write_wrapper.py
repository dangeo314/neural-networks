from datetime import datetime
import csv


filename = "../log/" + str(datetime.now()) + ".csv"

def write_row(obj = [""]):
    with open(filename, mode='a') as file1:
        writer1 = csv.writer(file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer1.writerow(obj)

