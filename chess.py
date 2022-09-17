# Imports
import cv2
import numpy as np
from PIL import Image
import pygame
import time
import random

pygame.mixer.init()

def play(sound):
    pygame.mixer.music.load("sounds/"+sound+".mp3")
    pygame.mixer.music.play()
SPECIAL_PIECES = ["Rook", "Knight", "Bishop", "Queen", "King", "Bishop", "Knight", "Rook"]  
ALL_PIECES = ["Pawn", "Rook", "Knight", "Bishop", "Queen", "King"]

images = []
white_images = []
for piece in ALL_PIECES:
    images.append(Image.fromarray(cv2.resize(cv2.imread("images/" + piece.lower() + ".png", -1), dsize=(100, 100), interpolation=cv2.INTER_LINEAR)))
    white_images.append(Image.fromarray(cv2.resize(cv2.imread("images/w" + piece.lower() + ".png", -1), dsize=(100, 100), interpolation=cv2.INTER_LINEAR)))

class Piece:

    def __init__(self, type, position):
        self.type = type
        self.position = position

    def toString(self):
        return self.type + " - " + str(tuple(self.position))

    def getType(self):
        return self.type

    def setType(self, new_type):
        self.type = new_type

    def getPosition(self):
        return self.position

    def setPosition(self, position):
        self.position = position

    def getValue(self):
        if self.type == "Pawn":
            return 1
        elif self.type == "Rook":
            return 5
        elif self.type == "Knight":
            return 3
        elif self.type == "Bishop":
            return 3
        elif self.type == "Queen":
            return 9
        elif self.type == "King":
            return 100
        else:
            return 0

class Board:

    def __init__(self):

        temp_white_pieces = []
        temp_black_pieces = []
        for i in range(8):
            temp_white_pieces.append(Piece("Pawn", (i, 6)))
            temp_black_pieces.append(Piece("Pawn", (i, 1)))
            
        for i in range(8):
            temp_white_pieces.append(Piece(SPECIAL_PIECES[i], (i, 7)))
            temp_black_pieces.append(Piece(SPECIAL_PIECES[i], (i, 0)))

        self.white_pieces = temp_white_pieces
        self.black_pieces = temp_black_pieces
        self.piece_selected = None
        self.illegal_move = None
        self.illegal_move_time = None
        self.start_time = time.time()
        self.choose_state = {"is_choosing": False, "piece_selected": None}

    def secondsPassed(self):
        return time.time() - self.start_time


    def getWhitePieces(self):
        return self.white_pieces

    def getBlackPieces(self):
        return self.black_pieces

    def getPiece(self, position):
        for piece in self.white_pieces:
            if piece.getPosition() == position:
                return piece
        for piece in self.black_pieces:
            if piece.getPosition() == position:
                return piece
        return Piece("Empty", position)

    def copy(self):
        new_board = Board()
        for piece in self.white_pieces:
            new_board.white_pieces.append(piece.copy())
        for piece in self.black_pieces:
            new_board.black_pieces.append(piece.copy())
        if self.piece_selected != None:
            new_board.piece_selected = self.piece_selected.copy()
        else:
            new_board.piece_selected = None
        new_board.illegal_move = self.illegal_move
        new_board.illegal_move_time = self.illegal_move_time
        new_board.start_time = self.start_time
        new_board.choose_state = dict(self.choose_state)
        return new_board

    def killPiece(self, piece):
        if piece in self.white_pieces:
            self.white_pieces.remove(piece)
        elif piece in self.black_pieces:
            self.black_pieces.remove(piece)

    def movePiece(self, piece, position):
        self.piece_selected = None
        if piece in self.white_pieces:
            self.white_pieces.remove(piece)
            piece.setPosition(position)
            self.white_pieces.append(piece)
        elif piece in self.black_pieces:
            self.black_pieces.remove(piece)
            piece.setPosition(position)
            self.black_pieces.append(piece)
        if piece.getType() == "Pawn" and position[1] == 0:
            self.choose_state["is_choosing"] = True
            self.choose_state["piece_selected"] = piece
        elif piece.getType() == "Pawn" and position[1] == 7:
            piece.setType("Queen")

    def selectPiece(self, piece):
        self.piece_selected = piece
      
    def availableMoves(self, piece):
        moves = []
        is_white = True
        p_type = piece.getType()

        if piece in self.black_pieces:
            is_white = False
        for i in range(8):
            for j in range(8):
                position = (i, j)
                if self.getPiece(position).getType() == "King":
                    continue
                if self.getPiece(position) == piece:
                    continue
                if is_white:
                    if self.getPiece(position) in self.white_pieces:
                        continue
                    if p_type == "Pawn":
                        if piece.getPosition()[1] == 6 and self.getPiece(position).getType()=="Empty" and self.getPiece((position[0], position[1]+1)).getType()=="Empty" and (position[0], position[1]+2) == piece.getPosition():
                            moves.append(position)
                        if self.getPiece(position).getType()=="Empty" and (position[0], position[1]+1) == piece.getPosition():
                            moves.append(position)
                        if ( (position[0]-1,position[1]+1) == piece.getPosition() or (position[0]+1,position[1]+1) == piece.getPosition() ) and self.getPiece(position).getType()!="Empty":
                            moves.append(position)
                else:
                    if self.getPiece(position) in self.black_pieces:
                        continue
                    if p_type == "Pawn":
                        if piece.getPosition()[1] == 1 and self.getPiece(position).getType()=="Empty" and self.getPiece((position[0], position[1]-1)).getType()=="Empty" and (position[0], position[1]-2) == piece.getPosition():
                            moves.append(position)
                        if self.getPiece(position).getType()=="Empty" and (position[0], position[1]-1) == piece.getPosition():
                            moves.append(position)
                        if ( (position[0]-1,position[1]-1) == piece.getPosition() or (position[0]+1,position[1]-1) == piece.getPosition() ) and self.getPiece(position).getType()!="Empty":
                            moves.append(position)

                if p_type == "Knight":
                    if (position[0]-2,position[1]-1) == piece.getPosition() or (position[0]-2,position[1]+1) == piece.getPosition() or (position[0]-1,position[1]-2) == piece.getPosition() or (position[0]-1,position[1]+2) == piece.getPosition() or (position[0]+1,position[1]-2) == piece.getPosition() or (position[0]+1,position[1]+2) == piece.getPosition() or (position[0]+2,position[1]-1) == piece.getPosition() or (position[0]+2,position[1]+1) == piece.getPosition():
                        moves.append(position)


                if p_type == "King":
                    king_position = piece.getPosition()
                    opposing_king_position = None
                    curr_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                    for move in curr_moves:
                        if is_white and self.getPiece((king_position[0]+move[0], king_position[1]+move[1])) in self.white_pieces:
                            curr_moves.remove(move)
                        elif not is_white and self.getPiece((king_position[0]+move[0], king_position[1]+move[1])) in self.black_pieces:
                            curr_moves.remove(move)
                    if is_white:
                        for new_piece in self.getBlackPieces():
                            if new_piece.getType() == "King":
                                opposing_king_position = new_piece.getPosition()
                    else:
                        for new_piece in self.getWhitePieces():
                            if new_piece.getType() == "King":
                                opposing_king_position = new_piece.getPosition()
                    possible_moves = list(curr_moves)
                    for move in possible_moves:
                        for newmove in possible_moves:
                            if (king_position[0]+move[0]+newmove[0], king_position[1]+move[1]+newmove[1]) == opposing_king_position:
                                curr_moves.remove(move)
                    for move in curr_moves:
                        if (king_position[0]+move[0], king_position[1]+move[1]) == position:
                            moves.append(position)

                if p_type == "Bishop":
                    bishop_position = piece.getPosition()

                    if abs(position[0]-bishop_position[0]) != abs(position[1]-bishop_position[1]):
                        continue
                    x = int(position[0]-bishop_position[0] < 0)*2-1
                    y = int(position[1]-bishop_position[1] < 0)*2-1

                    for k in range(1,abs(position[0]-bishop_position[0])):
                        if self.getPiece((position[0]+x*k, position[1]+y*k)).getType() != "Empty":
                            break
                    else:
                        moves.append(position)

                if p_type == "Rook":
                    rook_position = piece.getPosition()
                    if position[0] != rook_position[0] and position[1] != rook_position[1]:
                        continue
                    x = 0
                    y = 0
                    if (position[0] > rook_position[0]):
                        x = -1
                    elif (position[0] < rook_position[0]):
                        x = 1
                    else:
                        x = 0
                    if (position[1] > rook_position[1]):
                        y = -1
                    elif (position[1] < rook_position[1]):
                        y = 1
                    else:
                        y = 0
                    if y == 0:
                        for k in range(1,abs(position[0]-rook_position[0])):
                            if self.getPiece((position[0]+x*k, position[1])).getType() != "Empty":
                                break
                        else:
                            moves.append(position)
                    else:
                        for k in range(1,abs(position[1]-rook_position[1])):
                            if self.getPiece((position[0], position[1]+y*k)).getType() != "Empty":
                                break
                        else:
                            moves.append(position)

                if p_type == "Queen":
                    rook_position = piece.getPosition()
                    if not (position[0] != rook_position[0] and position[1] != rook_position[1]):
                        
                        x = 0
                        y = 0
                        if (position[0] > rook_position[0]):
                            x = -1
                        elif (position[0] < rook_position[0]):
                            x = 1
                        else:
                            x = 0
                        if (position[1] > rook_position[1]):
                            y = -1
                        elif (position[1] < rook_position[1]):
                            y = 1
                        else:
                            y = 0
                        if y == 0:
                            for k in range(1,abs(position[0]-rook_position[0])):
                                if self.getPiece((position[0]+x*k, position[1])).getType() != "Empty":
                                    break
                            else:
                                moves.append(position)
                        else:
                            for k in range(1,abs(position[1]-rook_position[1])):
                                if self.getPiece((position[0], position[1]+y*k)).getType() != "Empty":
                                    break
                            else:
                                moves.append(position)
                    if position not in moves:
                        bishop_position = piece.getPosition()
                        if abs(position[0]-bishop_position[0]) != abs(position[1]-bishop_position[1]):
                            continue
                        x = int(position[0]-bishop_position[0] < 0)*2-1
                        y = int(position[1]-bishop_position[1] < 0)*2-1

                        for k in range(1,abs(position[0]-bishop_position[0])):
                            if self.getPiece((position[0]+x*k, position[1]+y*k)).getType() != "Empty":
                                break
                        else:
                            moves.append(position)
                    
        return moves

board = Board()


class Move:
    def __init__(self, piece, position):
        self.piece = piece
        self.position = position
    
    def getPiece(self):
        return self.piece
    
    def getPosition(self):
        return self.position

class Bot:

    def __init__(self):
        self.moves_made = []

    def processMove(self):
        places = []
        for i in range(8):
            for j in range(8):
                places.append((i, j))
        random.shuffle(places)
        weights = []
        moves = []
        for i in range(64):
            # print(board.getPiece(places[i]).toString())
            piece = board.getPiece(places[i])
            if piece in board.getBlackPieces():
                # if len(board.availableMoves(piece)) > 0:
                for move in board.availableMoves(piece):
                    moves.append(Move(piece, move))
                    weights.append(board.getPiece(move).getValue())
        highest_val = max(weights)
        best_moves = []
        for i in range(len(moves)):
            if weights[i] == highest_val:
                best_moves.append(moves[i])
        return random.choice(best_moves)
        
            
np_board = np.zeros((8, 8, 3), np.uint8)
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        area_clicked = (int(x/100), int(y/100))
        curr_piece = board.getPiece(area_clicked)

        if board.choose_state["piece_selected"] != None:
            x_pos = board.choose_state["piece_selected"].getPosition()[0]
            types = ["Queen", "Bishop", "Rook", "Knight"]
            if area_clicked[0] == x_pos and area_clicked[1] < 4:
                board.choose_state["piece_selected"] = None
                board.choose_state["is_choosing"] = False
                board.getPiece((x_pos, 0)).setType(types[int(area_clicked[1])])

        elif curr_piece in board.getWhitePieces() and board.piece_selected == None:
            board.selectPiece(curr_piece)
        elif board.piece_selected != None and (area_clicked in board.availableMoves(board.piece_selected)):
            if board.getPiece((area_clicked)) in board.getWhitePieces():
                board.piece_selected = None
            else:
                if board.getPiece((area_clicked)).getType() == "Empty":
                    board.movePiece(board.piece_selected, area_clicked)
                    play("move")
                else:
                    board.killPiece(board.getPiece((area_clicked)))
                    board.movePiece(board.piece_selected, area_clicked)
                    play("capture")
                bot_move = bot.processMove()
                bot_pos = bot_move.getPosition()
                bot_piece = bot_move.getPiece()
                if board.getPiece((bot_pos)).getType() == "Empty":
                    board.movePiece(bot_piece, bot_pos)
                    play("move")
                else:
                    board.killPiece(board.getPiece((bot_pos)))
                    board.movePiece(bot_piece, bot_pos)
                    play("capture")

                board.movePiece(bot_move.getPiece(), bot_move.getPosition())
        elif board.piece_selected:
            play("illegal")
            board.illegal_move = area_clicked
            board.illegal_move_time = time.time()
            board.piece_selected = None

cv2.imshow("Chess", np_board)
cv2.setMouseCallback("Chess", mouse_click)


bot = Bot()
# Main Loop
while True:

    # Display Board
    np_board = np.zeros((8, 8, 3), np.uint8)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                np_board[i][j] = (208, 236, 235)
            else:
                np_board[i][j] = (86, 149, 119)
    np_board = cv2.resize(np_board, (800, 800), interpolation = cv2.INTER_AREA)
    if board.piece_selected != None:
        if (board.piece_selected.getPosition()[0]+board.piece_selected.getPosition()[1]) % 2 == 0:
            np_board[board.piece_selected.getPosition()[1] * 100 : (board.piece_selected.getPosition()[1] + 1) * 100, board.piece_selected.getPosition()[0] * 100 : (board.piece_selected.getPosition()[0] + 1) * 100] = (105, 247, 247)
        else:
            np_board[board.piece_selected.getPosition()[1] * 100 : (board.piece_selected.getPosition()[1] + 1) * 100, board.piece_selected.getPosition()[0] * 100 : (board.piece_selected.getPosition()[0] + 1) * 100] = (43, 203, 187)
        for block in board.availableMoves(board.piece_selected):
            color = (77, 135, 106)
            if (block[0]+block[1]) % 2 == 0:
                color = (189, 214, 214)
            if (board.getPiece(block).getType() == "Empty"):
                cv2.circle(np_board, (block[0]*100+50, block[1]*100+50), 18, color, -1)
            elif board.getPiece(block) in board.getBlackPieces():
                cv2.circle(np_board, (block[0]*100+50, block[1]*100+50), 45, color, 8)
    if board.illegal_move != None and time.time() - board.illegal_move_time < 0.3:
        if (board.illegal_move[0]+board.illegal_move[1]) % 2 == 0:
            np_board[board.illegal_move[1] * 100 : (board.illegal_move[1] + 1) * 100, board.illegal_move[0] * 100 : (board.illegal_move[0] + 1) * 100] = (105, 119, 246)
        else:
            np_board[board.illegal_move[1] * 100 : (board.illegal_move[1] + 1) * 100, board.illegal_move[0] * 100 : (board.illegal_move[0] + 1) * 100] = (43, 75, 186)
    temp_board = Image.fromarray(np_board)
    for troop in board.getWhitePieces():
        temp_board.paste(white_images[ALL_PIECES.index(troop.getType())], (troop.getPosition()[0]*100, troop.getPosition()[1]*100), white_images[ALL_PIECES.index(troop.getType())])
    for troop in board.getBlackPieces():
        temp_board.paste(images[ALL_PIECES.index(troop.getType())], (troop.getPosition()[0]*100, troop.getPosition()[1]*100), images[ALL_PIECES.index(troop.getType())])
    if board.choose_state and board.choose_state["piece_selected"] != None:
        x_axis = board.choose_state["piece_selected"].getPosition()[0]*100
        y_axis = 0
        temp_board = np.array(temp_board)
        temp_board[y_axis:y_axis+400, x_axis:x_axis+100] = (255, 255, 255)
        temp_board = Image.fromarray(temp_board)
        temp_board.paste(white_images[ALL_PIECES.index("Queen")], (x_axis, y_axis), white_images[ALL_PIECES.index("Queen")])
        temp_board.paste(white_images[ALL_PIECES.index("Bishop")], (x_axis, y_axis+100), white_images[ALL_PIECES.index("Bishop")])
        temp_board.paste(white_images[ALL_PIECES.index("Rook")], (x_axis, y_axis+200), white_images[ALL_PIECES.index("Rook")])
        temp_board.paste(white_images[ALL_PIECES.index("Knight")], (x_axis, y_axis+300), white_images[ALL_PIECES.index("Knight")])

    np_board = np.array(temp_board)
    cv2.imshow("Chess", np_board)
    cv2.waitKey(1)