import chess.pgn # For parsing pgn file
import numpy as np
import torch
import torch.nn as nn
import os, sys
import logging
import tqdm
import getopt

from io import StringIO
from tokenizers import Tokenizer
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import BertPreTrainedModel, RobertaModel
from datasets import Dataset, set_progress_bar_enabled

set_progress_bar_enabled(False)

MAX_TOKENS = 200
EVAL_BATCH_SIZE = 32

class RobertaForSequenceMultiTargetRegression(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.n_outputs = config.num_labels
        self.roberta = RobertaModel(config)
        self.regressor = RobertaRegressionHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        targets = None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        head_output = self.regressor(sequence_output)
        outputs = (head_output,) + outputs[2:]
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(head_output.view(-1,self.n_outputs), targets.view(-1,self.n_outputs))
        outputs = (loss,) + outputs
        return outputs  #  (preds), (hidden_states), (attentions)

    
class RobertaRegressionHead(nn.Module):
    """Head for sentence-level Regression tasks."""

    def __init__(self, config):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def read_one_game(game):
    result = game.headers["Result"]
    board = chess.Board()
    move_history = board.variation_san(game.mainline_moves()).split()
    # Any strings ending in . are numbers i.e. 1. 2. 3. 4., ... these are just "grammar" and don't provide any additional information and so are removed.
    move_history = " ".join(mv for mv in move_history if not mv[-1] == ".")
    move_history += " " + result
    return move_history
        
def run_prediction(opened_pgn, batch_size=16):
    game = chess.pgn.read_game(opened_pgn)
    ignore_missing_results = False
    game_moves = []
    while game is not None:
        if 'Result' in game.headers and game.headers['Result'] != "*":
            move_history = read_one_game(game)
            game_moves.append(move_history)
            game = chess.pgn.read_game(opened_pgn)
        else:
            if not ignore_missing_results:
                ignore_missing_results = True
                logging.info('For improved predictions, games should include the result under the \'Result\' header. Assuming games with missing results are draws...')
            game.headers['Result'] = '1/2-1/2'
    logging.info(f'Completed reading {len(game_moves)} games.')
    tokenizer = RobertaTokenizerFast.from_pretrained('./finetuned_model_full', max_len=MAX_TOKENS-1)
    model = RobertaForSequenceMultiTargetRegression.from_pretrained('./finetuned_model_full')

    def tokenization(batched_text):
        return tokenizer(batched_text['moves'], padding = True, truncation=True, )

    test_dict = {'moves': game_moves, 'targets': len(game_moves) * [[1,1]]}
    test_data = Dataset.from_dict(test_dict)
    test_data = test_data.map(tokenization, batched = True, batch_size=len(test_data))
    test_data.set_format('torch', columns=['attention_mask', 'input_ids', 'targets'])
    all_results = []
    for batch in range(1 + len(test_data) // batch_size):
        test_batch = test_data[batch * batch_size : (batch+1) * batch_size]
        outputs = model(input_ids = test_batch['input_ids'], attention_mask = test_batch['attention_mask'], targets = test_batch['targets'])
        preds = outputs[1].detach().cpu()
        all_results.append(preds)
    all_results = torch.cat(all_results, 0) if len(all_results)>1 else all_results[0]
    return all_results
    
def main(argv):
    pgn_file = ''
    output_file = None
    moves_str = ''
    quiet = False
    try:
        opts, args = getopt.getopt(argv,"hi:o:s:q",["help", "pgn_file","output_file", "from_string", "quiet"])
    except getopt.GetoptError:
       logging.warning('guess_elo.py -h -i <pgn_file> -o <outputfile> -s <from_string> -q')
       sys.exit(2)
    for opt, arg in opts:
       if opt in ('-h', '--help'):
           logging.warning('guess_elo.py -h -i <pgn_file> -o <outputfile> -s <from_string> -p')
           logging.info('-h, --help: \n     Print help menu.')
           logging.info("-i, --pgn_file: \n     str: Path to pgn file of chess games to evaluate ELOs.")
           logging.info("-o, --output_fiile: \n    str: Name of output (csv) file for writing ELOs. Default None.")
           logging.info("-s, --from_string: \n    str: String of moves in a game to evaluate ELOs.")
           logging.info("-q, --quiet: \n     bool: Supresses printing ELO predictions to terminal.")
           logging.info("One of -i, -s arguments required.")
           logging.info("Example: python guess_elo.py -s 1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6 9.Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. \
                         O-O-O Rd8 13. Rxd7 Rxd7 14.Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0")
           sys.exit()
       elif opt in ("-i", "--pgn_file"):
           pgn_file = arg
       elif opt in ("-o", "--output_file"):
           output_file = arg
       elif opt in ("-s", "--from_string"):
           moves_str = arg
       elif opt in ("-q", "--quiet"):
           quiet = True
    if not pgn_file and not moves_str:
        logging.warning("Invalid options. Much provide 1 of -i <pgn_file> or -s <from_string>. Run guess_elo.py -h for help.")
        sys.exit()
    if pgn_file and moves_str:
        logging.warning("Invalid options. Provide either option -i <pgn_file> or -s <from_string>, but not both. Run guess_elo.py -h for help.")
        sys.exit()
    pgn = open(pgn_file) if pgn_file else StringIO(moves_str)
    outputs = run_prediction(pgn).numpy()
    if not quiet:
        logging.info("Average ELO(s) : " + str(np.round(outputs.mean(axis=1))))
    if output_file:
        with open(output_file) as f:
            np.savetxt(outputs, output_file)
        
if __name__ == '__main__':
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    main(sys.argv[1:])
        
        
