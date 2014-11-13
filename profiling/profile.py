import sys
import subprocess

args = []

n = sys.argv[1].split('.')[1]
n = n.split('/')[1]
args.append('python')
args.append('-m')
args.append('cProfile')
args.append('-o')

stat_name = str(n) + '.pstats';
args.append(stat_name)

for i in range(1, len(sys.argv)):
    args.append(str(sys.argv[i]))


#print args
#quit()
print subprocess.check_output(args)

gargs = []
gargs.append('python')
gargs.append('./gprof2dot.py')
gargs.append('-f')
gargs.append('pstats')
gargs.append(stat_name)

#gargs.append('|')
#gargs.append('-Tpdf')
#gargs.append('-o')
#gargs.append(str(n) + '.pdf');
#./gprof2dot.py -f pstats output.pstats | dot -Tpdf -o output.pdf

p = subprocess.Popen(gargs, stdout=subprocess.PIPE)
out, err = p.communicate()

f = open(str(n) + '.dot', 'w')
f.write(out)
f.close()
