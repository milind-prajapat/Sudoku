import copy

def Cells(Grid):
    global Empty_Cells, Filled_Cells
    
    Empty_Cells = []
    Filled_Cells = []

    for i in range(9):
        for j in range(9):
            if Grid[i][j] == 0:
                Empty_Cells.append([i, j])
            else:
                Filled_Cells.append([i, j])

def isValid(Grid, x, y, Value):
    global X, Y
    
    for i in range(9):
        if Grid[i][y] == Value:
            return False
        if Grid[x][i] == Value:
            return False
        if Grid[X + int(i / 3)][Y + i % 3] == Value:
            return False
    
    return True

def Possibilities(Grid, x, y):
    global X, Y
    
    List = []
    X, Y = int(x / 3) * 3, int(y / 3) * 3

    for Value in range(1, 10):
        if isValid(Grid, x, y, Value):
            List.append(Value)
            
    return List 

def Constraints(Grid):
    global Empty_Cells, Filled_Cells

    List = [[0] * 9 for _ in range(9)]

    for x,y in Empty_Cells:
        List[x][y] = Possibilities(Grid, x, y)

    for x,y in Filled_Cells:
        List[x][y] = [Grid[x][y]]

    return List

def Check(L):
    Row_Check = False
    Col_Check = False
    Row = []
    Col = []

    for i in L:
        Row.append(any(i))

    for i in zip(*L):
        Col.append(any(i))

    R, C = 0, 0
    R_N, C_N = -1, -1

    for i in range(3):
        if Row[i]:
            R += 1
            R_N = i            
        if Col[i]:
            C += 1
            C_N = i

    if R == 1:
        Row_Check = True
    else:
        R_N = -1

    if C == 1:
        Col_Check = True
    else:
        C_N = -1
    
    return Row_Check, Col_Check, R_N, C_N

def Complete(L1, L2):
    for i in L1:
        if i not in L2:
            return False
    return True

def Partial(L1, L2):
    for i in L1:
        if i in L2:
            return True
    return False

def Update_Constraints(Grid, Constraint, Empty_Cells):
    isUpdated = False

    for x in range(3):
        for y in range(3):
            Elements = []

            for i in range(x * 3, x * 3 + 3):
                for j in range(y * 3, y * 3 + 3):
                    Elements.extend(Constraint[i][j])

            Row_Pairs = [[], [], []]
            Col_Pairs = [[], [], []]

            for Value in set(Elements):
                Pos = [[0] * 3 for _ in range(3)]

                for i in range(x * 3, x * 3 + 3):
                    for j in range(y * 3, y * 3 + 3):
                        if Value in Constraint[i][j]:
                            Pos[i % 3][j % 3] = 1

                Row_Check, Col_Check, R_N, C_N = Check(Pos)

                if Row_Check and Col_Check and [R_N + (3 * x), C_N + (3 * y)] not in Empty_Cells:
                    continue

                if Row_Check and Col_Check:
                    Constraint[R_N + (3 * x)][C_N + (3 * y)] = [Value]
                    Grid[R_N + (3 * x)][C_N + (3 * y)] = Value
                    Empty_Cells.remove([R_N + (3 * x), C_N + (3 * y)])

                if Row_Check:
                    if not Col_Check:
                        Row_Pairs[R_N].append(Value)

                    for i in range(0, y * 3):
                        if Value in Constraint[R_N + (3 * x)][i]:
                            Constraint[R_N + (3 * x)][i].remove(Value) 
                            
                            if not len(Constraint[R_N + (3 * x)][i]):
                                return -1
                            isUpdated = True

                    for i in range(y * 3 + 3, 9):
                        if Value in Constraint[R_N + (3 * x)][i]:
                            Constraint[R_N + (3 * x)][i].remove(Value)
                            
                            if not len(Constraint[R_N + (3 * x)][i]):
                                return -1
                            isUpdated = True

                if Col_Check:
                    if not Row_Check:
                        Col_Pairs[C_N].append(Value)

                    for i in range(0, x * 3):
                        if Value in Constraint[i][C_N + (3 * y)]:
                            Constraint[i][C_N + (3 * y)].remove(Value) 
                            
                            if not len(Constraint[i][C_N + (3 * y)]):
                                return -1
                            isUpdated = True

                    for i in range(x * 3 + 3, 9):
                        if Value in Constraint[i][C_N + (3 * y)]:
                            Constraint[i][C_N + (3 * y)].remove(Value) 
                            
                            if not len(Constraint[i][C_N + (3 * y)]):
                                return -1
                            isUpdated = True

            for R_N, R in enumerate(Row_Pairs):
                if len(R) == 2 or len(R) == 3:
                    Count = 0

                    for i in range(3):
                        if Complete(R,Constraint[R_N + (3 * x)][i + (3 * y)]):
                            Count += 1
                        elif Partial(R,Constraint[R_N + (3 * x)][i + (3 * y)]):
                            Count = 0
                            break

                    if Count == len(R):
                        for i in range(3):
                            if Complete(R, Constraint[R_N + (3 * x)][i + (3 * y)]):
                                Constraint[R_N + (3 * x)][i + (3 * y)] = R[:]

            for C_N, C in enumerate(Col_Pairs):
                if len(C) == 2 or len(C) == 3:
                    Count = 0

                    for i in range(3):
                        if Complete(C, Constraint[(3 * x) + i][C_N + (3 * y)]):
                            Count += 1
                        elif Partial(C, Constraint[(3 * x) + i][C_N + (3 * y)]):
                            Count = 0
                            break

                    if Count == len(C):
                        for i in range(3):
                            if Complete(C, Constraint[(3 * x) + i][C_N + (3 * y)]):
                                Constraint[(3 * x) + i][C_N + (3 * y)] = C[:]

    for x in range(9):
        Elements = []

        for y in range(9):
            Elements.extend(Constraint[x][y])

        for Value in set(Elements):
            Count = 0
            X, Y = -1,-1

            for y in range(9):
                if Value in Constraint[x][y]:
                    Count += 1
                    X, Y = x, y
                if Count > 1:
                    X, Y = -1,-1
                    break

            if Count == 1:
                if [X, Y] not in Empty_Cells:
                    continue

                Constraint[X][Y] = [Value]
                Grid[X][Y] = Value
                Empty_Cells.remove([X, Y])
                isUpdated = True
                
                for i in range(0, X):
                    if Value in Constraint[i][Y]:
                        Constraint[i][Y].remove(Value)
                        
                        if not len(Constraint[i][Y]):
                            return -1

                for i in range(X + 1, 9):
                    if Value in Constraint[i][Y]:
                        Constraint[i][Y].remove(Value) 
                        
                        if not len(Constraint[i][Y]):
                            return -1
    
                cX, cY = int(X / 3), int(Y / 3)

                for i in range(3):
                    for j in range(3):
                        if Value in Constraint[i + cX * 3][j + cY * 3] and (i + cX * 3,j + cY * 3) != (X, Y):
                            Constraint[i + cX * 3][j + cY * 3].remove(Value)
                            
                            if not len(Constraint[i + cX * 3][j + cY * 3]):
                                return -1

        Elements = []

        for y in range(9):
            Elements.extend(Constraint[y][x])

        for Value in set(Elements):
            Count = 0
            X, Y = -1, -1

            for y in range(9):
                if Value in Constraint[y][x]:
                    Count += 1
                    X, Y = y, x
                if Count > 1:
                    X, Y = -1, -1
                    break

            if Count == 1:
                if [X, Y] not in Empty_Cells:
                    continue

                Constraint[X][Y] = [Value]
                Grid[X][Y] = Value
                Empty_Cells.remove([X, Y])
                isUpdated = True
                
                for i in range(0, Y):
                    if Value in Constraint[X][i]:
                        Constraint[X][i].remove(Value)
                        
                        if not len(Constraint[X][i]):
                            return -1

                for i in range(Y + 1, 9):
                    if Value in Constraint[X][i]:
                        Constraint[X][i].remove(Value)
                        
                        if not len(Constraint[X][i]):
                            return -1
                
                cX, cY = int(X / 3), int(Y / 3)

                for i in range(3):
                    for j in range(3):
                        if Value in Constraint[i + cX * 3][j + cY * 3] and (i + cX * 3,j + cY * 3) != (X, Y):
                            Constraint[i + cX * 3][j + cY * 3].remove(Value)
                            
                            if not len(Constraint[i + cX * 3][j + cY * 3]):
                                return -1      
                            
    return isUpdated

def Update(x, y):
    global Grid_List, List, Empty_List
    
    while True:
        if Grid_List[-1][x][y] == 0:
            List.append(copy.deepcopy(List[-1]))
            Empty_List.append(copy.deepcopy(Empty_List[-1]))
            
            Value = List[-1][x][y][0] 
            Grid_List[-1][x][y] = Value
            List[-1][x][y] = [Value]
        else:
            Index = List[-1][x][y].index(Grid_List[-1][x][y]) + 1
            
            if Index == len(List[-1][x][y]):
                Grid_List.remove(Grid_List[-1])
                List.remove(List[-1])
                Empty_List.remove(Empty_List[-1])
                return False
            
            List.append(copy.deepcopy(List[-1]))
            Empty_List.append(copy.deepcopy(Empty_List[-1]))
            
            Value = List[-1][x][y][Index]
            Grid_List[-1][x][y] = Value
            List[-1][x][y] = [Value]

        Check = False

        for i in range(0, y):
            if Value in List[-1][x][i]:
                List[-1][x][i].remove(Value)
                
                if not len(List[-1][x][i]):
                    Check = True
                    break
        if Check:
            List.remove(List[-1])
            Empty_List.remove(Empty_List[-1])
            continue

        for i in range(y + 1, 9):
            if Value in List[-1][x][i]:
                List[-1][x][i].remove(Value)
                
                if not len(List[-1][x][i]):
                    Check = True
                    break
                
        if Check:
            List.remove(List[-1])
            Empty_List.remove(Empty_List[-1])
            continue

        for i in range(0, x):
            if Value in List[-1][i][y]:
                List[-1][i][y].remove(Value)   
                
                if not len(List[-1][i][y]):
                    Check = True
                    break

        if Check:
            List.remove(List[-1])
            Empty_List.remove(Empty_List[-1])
            continue

        for i in range(x + 1, 9):
            if Value in List[-1][i][y]:
                List[-1][i][y].remove(Value)
                
                if not len(List[-1][i][y]):
                    Check = True
                    break

        if Check:
            List.remove(List[-1])
            Empty_List.remove(Empty_List[-1])
            continue

        cX, cY = int(x / 3), int(y / 3)

        for i in range(3):
            for j in range(3):
                if Value in List[-1][i + cX * 3][j + cY * 3] and (i + cX * 3,j + cY * 3) != (x, y):
                    List[-1][i + cX * 3][j + cY * 3].remove(Value)
                    
                    if not len(List[-1][i + cX * 3][j + cY * 3]):
                        Check = True
                        break

        if Check:
            List.remove(List[-1])
            Empty_List.remove(Empty_List[-1])
            continue

        Grid_List.append(copy.deepcopy(Grid_List[-1]))

        while True:
            Val = Update_Constraints(Grid_List[-1], List[-1], Empty_List[-1])
            
            if not Val:
                return True
            
            if Val == -1:
                Grid_List.remove(Grid_List[-1])
                List.remove(List[-1])
                Empty_List.remove(Empty_List[-1])
                break

def is_Sudoku_Valid(Grid):
    for i in range(9):
        Row = []
        Col = []
        
        for j in range(9):
            if Grid[i][j] != 0:
                Row.append(Grid[i][j])
            
            if Grid[j][i] != 0:
                Col.append(Grid[j][i])
        
        if len(Row) != len(set(Row)):
            return False
        
        if len(Col) != len(set(Col)):
            return False

    for x in range(3):
        for y in range(3):
            Cell = []

            for i in range(x * 3, x * 3 + 3):
                for j in range(y * 3, y * 3 + 3):
                    if Grid[i][j] != 0:
                        Cell.append(Grid[i][j])
            if len(Cell) != len(set(Cell)):
                return False
               
    return True

def Solve(Grid):
    global Grid_List, Empty_List, List, Empty_Cells
    
    if not is_Sudoku_Valid(Grid):
        return None, None

    try:
        Cells(Grid)
        E_Cells = copy.deepcopy(Empty_Cells)
        Constraint = Constraints(Grid)

        while(Update_Constraints(Grid, Constraint, E_Cells)):
            pass

        List = [Constraint]
        Empty_List = [E_Cells]
        Grid_List = [Grid]

        while len(Empty_List[-1]):
            x, y = Empty_List[-1][0]
            Update(x, y)
    except:
        return None, None
        
    return Grid_List[-1], Empty_Cells