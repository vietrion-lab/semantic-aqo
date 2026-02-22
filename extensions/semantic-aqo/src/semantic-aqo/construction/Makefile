CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2 -g
LDFLAGS = -lm

SRC = vectorize_sql.c
OBJ = vectorize_sql.o

TEST_SRC = test_vectorize.c
TEST_BIN = test_vectorize

.PHONY: all test clean

all: $(TEST_BIN)

$(OBJ): $(SRC) vectorize_sql.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_BIN): $(TEST_SRC) $(OBJ) vectorize_sql.h
	$(CC) $(CFLAGS) $(TEST_SRC) $(OBJ) -o $@ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(OBJ) $(TEST_BIN)
