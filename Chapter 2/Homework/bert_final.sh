python imdb.py --epochs 2 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 42   --group bert-ep2-lr8e-5 --wandb
sleep 30
python imdb.py --epochs 2 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 314  --group bert-ep2-lr8e-5 --wandb
sleep 30
python imdb.py --epochs 2 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 1618 --group bert-ep2-lr8e-5 --wandb
sleep 30
python imdb.py --epochs 3 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 42   --group bert-ep3-lr8e-5 --wandb
sleep 30
python imdb.py --epochs 3 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 314  --group bert-ep3-lr8e-5 --wandb
sleep 30
python imdb.py --epochs 3 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 1618 --group bert-ep3-lr8e-5 --wandb
sleep 30
python imdb.py --epochs 4 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 42   --group bert-ep4-lr8e-5 --wandb
sleep 30
python imdb.py --epochs 4 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 314  --group bert-ep4-lr8e-5 --wandb
sleep 30
python imdb.py --epochs 4 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 1618 --group bert-ep4-lr8e-5 --wandb
sleep 30

python imdb.py --epochs 2 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --seed 42   --group bert-ep2-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 2 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --seed 314  --group bert-ep2-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 2 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --seed 1618 --group bert-ep2-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 3 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --seed 42   --group bert-ep3-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 3 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --seed 314  --group bert-ep3-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 3 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --seed 1618 --group bert-ep3-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 4 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --seed 42   --group bert-ep4-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 4 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --seed 314  --group bert-ep4-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 4 --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --seed 1618 --group bert-ep4-lr8e-5-adan --wandb
sleep 30

python imdb.py --epochs 1 --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 42   --group bert-ep2-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 1 --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 314  --group bert-ep2-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 1 --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 1618 --group bert-ep2-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 2 --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 42   --group bert-ep2-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 2 --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 314  --group bert-ep2-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 2 --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 1618 --group bert-ep2-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 3 --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 42   --group bert-ep3-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 3 --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 314  --group bert-ep3-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 3 --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --seed 1618 --group bert-ep3-lr8e-5 --wandb


python imdb.py --epochs 1 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 42   --group deberta-ep1-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 1 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 314  --group deberta-ep1-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 1 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 1618 --group deberta-ep1-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 42   --group deberta-ep2-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 314  --group deberta-ep2-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 1618 --group deberta-ep2-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 42   --group deberta-ep3-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 314  --group deberta-ep3-lr1e-4 --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 1618 --group deberta-ep3-lr1e-4 --wandb
sleep 30

python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 42   --group deberta-ep2-lr5e-5 --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 314  --group deberta-ep2-lr5e-5 --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 1618 --group deberta-ep2-lr5e-5 --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 42   --group deberta-ep3-lr5e-5 --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 314  --group deberta-ep3-lr5e-5 --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 1618 --group deberta-ep3-lr5e-5 --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 42   --group deberta-ep4-lr5e-5 --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 314  --group deberta-ep4-lr5e-5 --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --batch-size 40 --seed 1618 --group deberta-ep4-lr5e-5 --wandb
sleep 30

python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 42   --group deberta-ep2-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 314  --group deberta-ep2-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 1618 --group deberta-ep2-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 42   --group deberta-ep3-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 314  --group deberta-ep3-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 1618 --group deberta-ep3-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 42   --group deberta-ep4-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 314  --group deberta-ep4-lr8e-5-adan --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 5e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 1618 --group deberta-ep4-lr8e-5-adan --wandb


python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 42   --group deberta-ep2-lr8e-5r-adan --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 314  --group deberta-ep2-lr8e-5r-adan --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 1618 --group deberta-ep2-lr8e-5r-adan --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 42   --group deberta-ep3-lr8e-5r-adan --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 314  --group deberta-ep3-lr8e-5r-adan --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 1618 --group deberta-ep3-lr8e-5r-adan --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 42   --group deberta-ep4-lr8e-5r-adan --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 314  --group deberta-ep4-lr8e-5r-adan --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 8e-5 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 1618 --group deberta-ep4-lr8e-5r-adan --wandb
sleep 30

python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 42   --group deberta-ep2-lr1e-4-adan --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 314  --group deberta-ep2-lr1e-4-adan --wandb
sleep 30
python imdb.py --epochs 2 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 1618 --group deberta-ep2-lr1e-4-adan --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 42   --group deberta-ep3-lr1e-4-adan --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 314  --group deberta-ep3-lr1e-4-adan --wandb
sleep 30
python imdb.py --epochs 3 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 1618 --group deberta-ep3-lr1e-4-adan --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 42   --group deberta-ep4-lr1e-4-adan --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 314  --group deberta-ep4-lr1e-4-adan --wandb
sleep 30
python imdb.py --epochs 4 --model microsoft/deberta-v3-base --learning-rate 1e-4 --train-subset -1 --eval-subset -1 --no-early-stopping --opt adan --batch-size 32 --seed 1618 --group deberta-ep4-lr1e-4-adan --wandb
