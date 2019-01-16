git ls-files | grep -E '^CMakeLists.txt|^oz\/.*\.(py|cpp|h)$' | xargs sed -i'' 's/leduk/leduc/g; s/Leduk/Leduc/g'

for f in `git ls-files | grep -i leduk`; do
	git mv $f `echo $f | sed s/leduk/leduc/g`
done
