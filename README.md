Implementation of various RAT approaches described in [Hoang et al., Improving Retrieval Augmented Neural Machine Translation by
Controlling Source and Fuzzy-Match Interactions, Findings of EACL 2023](https://aclanthology.org/2023.findings-eacl.22.pdf)

and

a new approach (RAT-SEPD), where D_ENC shares the self-attention and ffn weights with DEC

## Approaches
| Model    | Module | INPUT                                                  |
| :------- | :----- | :----------------------------------------------------- |
| Baseline | ENC    | SRC                                                    |
|          | DEC    | TGT                                                    |
| RAT-CAT  | ENC    | SRC + tCtx_1 + ... + tCtx_k                            |
|          | DEC    | TGT                                                    |
| RAT-SEP  | ENC    | SRC                                                    |
|          | ENC2   | tCtx_1                                                 |
|          |        |   ...                                                  |
|          |        | tCtx_k                                                 |
|          | DEC    | TGT                                                    |
| RAT-SI   | ENC    | SRC                                                    |
|          |        | tCtx_1 + SRC                                           |
|          |        |   ...                                                  |
|          |        | tCtx_k + SRC                                           |
|          | DEC    | TGT                                                    |
| RAT-SEPD | ENC    | SRC                                                    |
|          | D_ENC  | tCtx_1                                                 |
|          |        |   ...                                                  |
|          |        | tCtx_k                                                 |
|          | DEC    | TGT                                                    |

RAT-CAT and RAT-SI utilizes an additional token '｟SENSEP｠' to mark the boundaries between sentences in the encoder input.

Note that RAT-SI is implemented in a slightly different way than the paper (tCtx_k + SRC instead of SRC + tCtx_k),
but the result should be identical.


## Commandline examples
```
onmt-main --model src/BaselineTransformer.py --config config/wmt14_ende_BaselineTransformer.yml --auto_config train --with_eval > run/wmt14_ende_BaselineTransformer.log 2>&1

onmt-main --model src/BaselineTransformer.py --config config/wmt14_ende_RatCATTransformer.yml --auto_config train --with_eval > run/wmt14_ende_RatCATTransformer.log 2>&1

onmt-main --model src/RatSEPTransformer.py --config config/wmt14_ende_RatSEPTransformer.yml --auto_config train --with_eval > run/wmt14_ende_RatSEPTransformer.log 2>&1

onmt-main --model src/RatSITransformer.py --config config/wmt14_ende_RatSITransformer.yml --auto_config train --with_eval > run/wmt14_ende_RatSITransformer.log 2>&1

onmt-main --model src/RatSEPDTransformer.py --config config/wmt14_ende_RatSEPDTransformer.yml --auto_config train --with_eval > run/wmt14_ende_RatSEPDTransformer.log 2>&1
```

Using top 3 contexts:
```
onmt-main --model src/RatSITransformer_top3.py --config config/wmt14_ende_RatSITransformer_top3.yml --auto_config train --with_eval > run/wmt14_ende_RatSITransformer_top3.log 2>&1 &
```