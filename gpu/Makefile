# Makefile

CC=nvcc
CFLAGS=--ptxas-options=-v --compiler-options '-fPIC'
TARGET=culib/libDeviceQuery.so
SRC=culib/deviceQuery.cpp

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) --shared $(SRC)


clean:
	rm -f $(TARGET)
