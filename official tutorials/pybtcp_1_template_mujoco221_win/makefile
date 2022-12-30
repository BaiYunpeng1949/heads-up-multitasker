
#MAC
#COMMON=-O2 -I../../include -L../../bin -mavx -pthread
#LIBS = -w -lmujoco200 -lglfw.3
#CC = gcc

#LINUX
#COMMON=-O2 -I../../include -L../../bin -mavx -pthread -Wl,-rpath,'$$ORIGIN'
#LIBS = -lmujoco200 -lGL -lm -lglew ../../bin/libglfw.so.3
#CC = gcc

#WINDOWS
COMMON=/O2 /MT /EHsc /arch:AVX /I../../include /Fe../../bin/
LIBS = ../../lib/glfw3dll.lib  ../../lib/mujoco.lib 
CC = cl

ROOT = main

all:
	$(CC) $(COMMON) main.c $(LIBS) -o $(ROOT)

main.o:
	$(CC) $(COMMON) -c main.c

clean:
	rm *.o ../../bin/$(ROOT)