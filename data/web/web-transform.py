import csv, sys, getopt
import numpy as np
import scipy.io as sio

def convert_2label(number):
    return "%03d" % int(bin(number)[2:])

def convert_5label(number):
    add = ""
    if number == 1:
        add = "00001"
    elif number == 2:
        add = "00010"
    elif number == 3:
        add = "00100"
    elif number == 4:
        add = "01000"
    elif number == 5:
        add = "10000"
    return add

def convert_to_mat(input, output):
    choice = int(raw_input("Enter a choice:\n 1. One-hot encoding\n 2. Binary Encoding\n 3. Real-values\t"))
    with open(input, 'rb') as tsvin:
        spamreader = csv.reader(tsvin, delimiter=',')

        final_labels = []
        intermediate = []
        count = 1
        for row in spamreader:
            num = int(row[2])
            if choice == 1:
                intermediate.append(convert_5label(num))
            elif choice == 2:
                intermediate.append(convert_2label(num))
            elif choice == 3:
                intermediate.append(num)
            count += 1
            if count == 7:
                if choice == 1 or choice == 2:
                    intermediate = ''.join(intermediate)
                    intermediate = list(intermediate)
                intermediate = map(int, intermediate)
                final_labels.append(intermediate)
                intermediate = []
                count = 1
                continue

        final_labels = np.array(final_labels)
        sio.savemat(output, {'data': final_labels})

if __name__ == "__main__":
    infile = ''
    outfile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["in_file=", "out_file="])
    except getopt.GetoptError:
        print 'python web-transform.py -i <inputfile> -o <outputfile>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print 'python web-transform.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--in_file"):
            infile = arg
        elif opt in ("-o", "--out_file"):
            outfile = arg

    # convert to data
    convert_to_mat(infile, outfile)
