'''
Speed tests to improve the speed
of the chess engine.
'''
import main
import piece
import time
class tests:
    def test(depth):
        main.negamax(main.initial_board, depth, float('-inf'), float('inf'), 1, None, {})

    def test0():
        piece.evaluate(main.initial_board)

    def test1():
        5 << 2

    def test2():
        x = {1: 2, 3: 4}
        x[1]

    def test3():
        piece.evaluate(main.initial_board)

    def test4():
        x = 1 + 1
        if x == 2:
            return
        else:
            return


start = time.time()
for _ in range(1523):
    tests.test1()
end = time.time()
print(end-start)


