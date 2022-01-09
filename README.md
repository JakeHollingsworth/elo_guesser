In this notebook, we begin training a transformer to predict the ELO rating of 2 participants in a chess game given the move order. ELO ratings attempt to quantify and rank the ability of chess players; they are determined by the lichess ELO rating system (Glicko 2 rating system: https://lichess.org/page/rating-systems).

We use games selected from a single month of the lichess open database (https://database.lichess.org/). Only rapid time control games are included. Data is stored in .pgn format, a standard file format for recording chess games. For each game in the .pgn file, we extract the ELO of the white and black player, the move order given in algebraic notation (https://en.wikipedia.org/wiki/Algebraic_notation_(chess)) and the result of the game ("1-0" for white win, "1/2-1/2" for a draw, and "0-1" for black win) which is concatenated at the end of the move history. These are then written to a .csv file for easier processing.

We train a RoBERTa model from the Hugging Face library (https://huggingface.co/docs/transformers/index) on the bidirectional (dynamic) masked language modelling task in order to get an informative fixed size representation of the move history (hidden state representation of \<s> token). To our knowledge, no existing transformer model exists which has been trained on a large corpus of chess games in algebraic notation, and so transfer learning is not available and training begins with a random initialization of weights. After completing training on the masked language model task, we fine-tune the model to solve the particular problem of predicting player elo ratings from the move history. We organize our models in this way to mimic the organization of standard NLP applications, where a large pretraining step is done on the masked language model task, which is then fine-tuned for a particular use case.

Inspiration for this project is two-fold. First, the task itself has grown popular among notable players in the chess community (IM Levy Rozman: https://www.youtube.com/watch?v=0baCL9wwJTA&list=PLBRObSmbZluRiGDWMKtOTJiLy3q0zIfd7, GM Hikaru Nakamura: https://www.youtube.com/watch?v=HEU2Rkjs-RI). This library thus offers a machine learning tool to be used for evaluation of human predictions. 

A second use is to provide a metric for per-game performance of a player (alternative to computer accuracy). A players personal ELO rating only serves as an aggregate metric of performance across all of their games. The player performance for an individual game may be analyzed using a chess engine (i.e. percentage of moves played in top N suggestions by a chess engine), but this is not ideal as computers often suggest moves that are good due to reasons imperceptible to even the best human players. That is, computer accuracy is determined by the limits of computer ability. Thus, computer accuracy will tend to be lower for games where positions are highly complex (and vice versa), though player performance may not meaningfully differ. Rather than comparing to a chess engine, the RoBERTa model trained here evaluates the players performance based on the games of human players across all ELO ratings, and so provides a fairer metric. The predicted ELO rating is also arguably more interpretable than computer accuracy.

This project is still in progress.

Examples:

Opera Game: 
python guess_elo.py -s "1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6 9.Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7 14.Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0"
Returns: 1617

Scholar's Mate: 
python guess_elo.py -s "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#"
Returns: 1145

Gold Coin Game (struggles with very high elo games):
python guess_elo.py -s "1. d4 e6 2. e4 d5 3. Nc3 c5 4. Nf3 Nc6 5. exd5 exd5 6. Be2 Nf6 7. O-O Be7 8. Bg5 O-O 9. dxc5 Be6 10. Nd4 Bxc5 11. Nxe6 fxe6 12. Bg4 Qd6 13. Bh3 Rae8 14. Qd2 Bb4 15. Bxf6 Rxf6 16. Rad1 Qc5 17. Qe2 Bxc3 18. bxc3 Qxc3 19. Rxd5 Nd4 20. Qh5 Ref8 21. Re5 Rh6 22. Qg5 Rxh3 23. Rc5 Qg3 0-1"

Draw line in the Berlin:
python guess_elo.py -i berlin_draw_line.pgn
