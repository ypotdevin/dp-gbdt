COMPILER=g++
CFLAGS=-c -Wall -std=c++11 -g
#CFLAGS=-c -Wall -std=c++11 -Ofast -g
#CFLAGS=-c -Wall -std=c++11 -Ofast
SOURCES=$(wildcard src/*.cpp src/gbdt/*.cpp)
INC=-I./include/ # -L ./src/ -lfmt

$(shell mkdir -p objs)
$(shell mkdir -p objs/gbdt)

run: $(SOURCES:src/%.cpp=objs/%.o)
	$(COMPILER) -g -pthread $^ -o $@
#	$(COMPILER) -pthread $^ -o $@

clean:
	rm -r objs
	touch run && rm run

asm:
	make clean
	make CFLAGS="-c -std=c++11 -O1 -save-temps=obj"

objs/%.o : src/%.cpp
	$(COMPILER) $(CFLAGS) $(INC) $< -o $@
objs/gbdt/%.o : src/gbdt/%.cpp
	$(COMPILER) $(CFLAGS) $(INC) $< -o $@
