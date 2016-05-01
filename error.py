import csv
import json

def dog_results(filename):
    results = json.load(open(filename))
    rCount = [0,0,0,0]
    for a in results:
        if int(results[a]) == 1:
            rCount[0] += 1
        if int(results[a]) == 2:
            rCount[1] += 1
        if int(results[a]) == 3:
            rCount[2] += 1
        if int(results[a]) == 4:
            rCount[3] += 1

    # print rCount

    with open("dog-output.csv") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # from the original file
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

    # print aCount
    final_result = 0
    for i in xrange(4):
        final_result += abs(aCount[i] - rCount[i])

    print filename, float(final_result)/807

def web_results(filename):
    results = json.load(open(filename))
    rCount = [0,0,0,0,0]
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

    # print rCount

    with open("web-output.tsv") as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        # from the original file
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

    # print aCount
    final_result = 0
    for i in xrange(5):
        final_result += abs(aCount[i] - rCount[i])

    print filename, float(final_result)/2357


# print "Web Results"
web_results('results5vh.txt')
web_results('results3vh.txt')
web_results('results1v3h.txt')
web_results('results1v5h.txt')

# print "Dog Results"
dog_results('results4vh.txt')
dog_results('results2vh.txt')
dog_results('results1v2h.txt')
dog_results('results1v4h.txt')
