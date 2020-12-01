objects = main.o
cc = g++-7

compile: pagerank.o main.o
	$(cc) -g pagerank.o main.o -o main -fopenmp

pagerank.o: PageRank.h PageRank.cpp
	$(cc) -c -g PageRank.cpp -o pagerank.o -fopenmp
main.o: PageRank.h PageRank.cpp main.cpp
	$(cc) -c -g main.cpp -o main.o -fopenmp

.PHONY : clean
clean :
	rm *.o
	rm main
	rm -rf main.dSYM