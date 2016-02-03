all: run report clean
	
run: TINY_MNIST.npz
	rm -rf results
	mkdir results
	python KNN_Classification.py < inputs/task1
	python KNN_Classification.py < inputs/task2
	python LinearRegression.py < inputs/task3
	python LinearRegression.py < inputs/task4
	python LR_Classification.py < inputs/task5
	python LR_Classification.py < inputs/task6
	python LR_Classification.py < inputs/task7
	
report:
	pdflatex report.tex
	pdflatex report.tex
	
TINY_MNIST.npz:
	echo TINY_MNIST dataset not available
	false;

clean:
	rm -rf results *.aux *.log *.out