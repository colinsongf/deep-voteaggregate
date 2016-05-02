import csv
import json, sys, getopt

def dog_results(filename):
    results = json.load(open(filename))
    rCount = [0,0,0,0]
    # from the results, get the count
    for a in results:
        if int(results[a]) == 1:
            rCount[0] += 1
        if int(results[a]) == 2:
            rCount[1] += 1
        if int(results[a]) == 3:
            rCount[2] += 1
        if int(results[a]) == 4:
            rCount[3] += 1


    with open("data/dog/dog-output.csv") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # from the original file, get the count
        aCount = [0,0,0,0]
        for row in spamreader:
            ground_truth = int(row[1])
            if row[0] in results:
                if ground_truth == 0:
                    aCount[0] += 1
                if ground_truth == 1:
                    aCount[1] += 1
                if ground_truth == 2:
                    aCount[2] += 1
                if ground_truth == 3:
                    aCount[3] += 1

    # get the final L0 count
    final_result = 0
    for i in xrange(4):
        final_result += abs(aCount[i] - rCount[i])

    # caculate accuracy
    acc = round(float(final_result)/807,2)
    if acc > 1:
        print filename, 1.00
    else:
        print acc

def web_results(filename):
    results = json.load(open(filename))
    rCount = [0,0,0,0,0]
    # from the original file, get the count
    for a in results:
        if int(results[a]) == 1:
            rCount[0] += 1
        if int(results[a]) == 2:
            rCount[1] += 1
        if int(results[a]) == 3:
            rCount[2] += 1
        if int(results[a]) == 4:
            rCount[3] += 1
        if int(results[a]) == 5:
            rCount[4] += 1

    with open("data/web/web-output.tsv") as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        # from the original file, get the count
        aCount = [0,0,0,0,0]
        for row in spamreader:
            ground_truth = int(row[1])
            if row[0] in results:
                if ground_truth == 1:
                    aCount[0] += 1
                if ground_truth == 2:
                    aCount[1] += 1
                if ground_truth == 3:
                    aCount[2] += 1
                if ground_truth == 4:
                    aCount[3] += 1
                if ground_truth == 5:
                    aCount[4] += 1

    # get the final L0 count
    final_result = 0
    for i in xrange(5):
        final_result += abs(aCount[i] - rCount[i])

    # caculate accuracy
    acc = round(float(final_result)/2357,2)
    if acc > 1:
        print filename, 1.00
    else:
        print acc


if __name__ == "__main__":
    # file input
    infile = ''
    option = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["in_file=", "options="])
    except getopt.GetoptError:
        print 'python error.py -i <inputfile> -o <options>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print 'python error.py -i <inputfile> -o <options>'
            print '<options> --> 1 for web and 2 for dog'
            sys.exit()
        elif opt in ("-i", "--in_file"):
            infile = arg
        elif opt in ("-o", "--options"):
            option = arg

    # run the error modules
    if int(option) == 1:
        web_results(infile)
    elif int(option) == 2:
        dog_results(infile)


# print "Web Results"
# web_results('results/web/results5vh.txt')

# print "Dog Results"
# dog_results('results/dog/results4vh.txt')
