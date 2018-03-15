#!/usr/bin/bash

mkdir casict2015_clean80
# upload the files.

cd casict2015_clean80


cp ../casict2015/train.ipa train.ipa
cp ../casict2015/train.syl train.syl

cp ../casict2015/dev.ipa dev.ipa
cp ../casict2015/dev.syl dev.syl

cp ../test_dir/test.ipa test.ipa
cp ../test_dir/test.syl test.syl



OUTPUT_DIR=casict2015_clean80

if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

# Tokenize data
for f in ${OUTPUT_DIR}/*.ipa; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l zh -threads 8 < $f > ${f%.*}.tok.ipa
done

for f in ${OUTPUT_DIR}/*.syl; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.syl
done


# Clean all corpora
# removes empty lines
# removes redundant space characters
# drops lines (and their corresponding lines), that are empty, too short, too long or violate the 9-1 sentence ratio limit of GIZA++
for f in ${OUTPUT_DIR}/*.syl; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase ipa syl "${fbase}.clean" 1 80
done


for f in ${OUTPUT_DIR}/*.syl; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n-count-lines.perl $fbase ipa syl "${fbase}.clean" ${OUTPUT_DIR}/line_retained.txt 1 80
done

# For getting which lines are cleaned.





# Generate Subword Units (BPE)
# Clone Subword NMT
if [ ! -d "${OUTPUT_DIR}/subword-nmt" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${OUTPUT_DIR}/subword-nmt"
fi

# Learn Shared BPE
for merge_ops in 4000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.clean.ipa" "${OUTPUT_DIR}/train.tok.clean.syl" | \
    ${OUTPUT_DIR}/subword-nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"
#
  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in ipa syl; do
    for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.clean.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${OUTPUT_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done
#
  # Create vocabulary file for BPE
  echo -e "<unk>\n<s>\n</s>" > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"
  cat "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.syl" "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.ipa" | \
    ${OUTPUT_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' >> "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"
#
done

# Duplicate vocab file with language suffix
cp "${OUTPUT_DIR}/vocab.bpe.4000" "${OUTPUT_DIR}/vocab.bpe.4000.ipa"
cp "${OUTPUT_DIR}/vocab.bpe.4000" "${OUTPUT_DIR}/vocab.bpe.4000.syl"

echo "All done."



./run.pl casict2015_clean80_v3.log CUDA_VISIBLE_DEVICES=2 python -m nmt.nmt \
    --src=ipa --tgt=syl \
    --vocab_prefix=casict2015/vocab.bpe.4000  \
    --train_prefix=casict2015/train.tok.clean.bpe.4000 \
    --dev_prefix=casict2015/dev.tok.clean.bpe.4000  \
    --test_prefix=casict2015/test.tok.clean.bpe.4000 \
    --out_dir=nmt_model_standard_hparams_casict2015_clean80_v3 \
    --hparams_path=nmt/standard_hparams/casict2015_v3.json
######
######
./run.pl CUDA_VISIBLE_DEVICES=1 python -m nmt.nmt \
    --src=ipa --tgt=syl \
    --vocab_prefix=casict2015/vocab.bpe.4000  \
    --train_prefix=casict2015/train.tok.clean.bpe.4000 \
    --dev_prefix=casict2015/dev.tok.clean.bpe.4000  \
    --test_prefix=casict2015/test.tok.clean.bpe.4000 \
    --out_dir=nmt_model_standard_hparams_casict2015_iwslt15 \
    --hparams_path=nmt/standard_hparams/iwslt15.json
# dev bleu 89.6, test bleu 33.4
######

./run.pl casict2015_v3.log CUDA_VISIBLE_DEVICES=1 python -m nmt.nmt \
    --src=ipa --tgt=syl \
    --vocab_prefix=casict2015/vocab.bpe.4000  \
    --train_prefix=casict2015/train.tok.clean.bpe.4000 \
    --dev_prefix=casict2015/dev.tok.clean.bpe.4000  \
    --test_prefix=casict2015/test.tok.clean.bpe.4000 \
    --out_dir=nmt_model_standard_hparams_casict2015_v3 \
    --hparams_path=nmt/standard_hparams/casict2015_v3.json
# 
######

./run.pl casict2015_wmt16.log CUDA_VISIBLE_DEVICES=1 python -m nmt.nmt \
    --src=ipa --tgt=syl \
    --vocab_prefix=casict2015/vocab.bpe.4000  \
    --train_prefix=casict2015/train.tok.clean.bpe.4000 \
    --dev_prefix=casict2015/dev.tok.clean.bpe.4000  \
    --test_prefix=casict2015/test.tok.clean.bpe.4000 \
    --out_dir=nmt_model_standard_hparams_casict2015_wmt16 \
    --hparams_path=nmt/standard_hparams/wmt16.json
#


######
# inference
CUDA_VISIBLE_DEVICES=0 python -m nmt.nmt \
    --out_dir=nmt_model_standard_hparams_casict2015 \
    --inference_input_file=casict2015/test.tok.clean.bpe.4000.ipa \
    --inference_output_file=output_infer_casict2015
#
CUDA_VISIBLE_DEVICES=0 python -m nmt.nmt \
    --out_dir=nmt_model_standard_hparams_casict2015_v3 \
    --inference_input_file=casict2015/test.tok.clean.bpe.4000.ipa \
    --inference_output_file=output_infer_casict2015_v3
#








cd /home/fut/work_tensorflow/nmt2/nmt/test_dir
wc -l ali.phones.txt
wc -l test.char.en.syl.no-multi.txt

cat ali.phones.txt | awk '{print $1;}' > ali.phones.id.txt
cat test.char.en.syl.no-multi.txt | awk '{print $1;}' > test.syl.id.txt
# test.char.en.syl.no-multi.txt中多了两句，直接删掉， 得到test.syl.txt
# 20041027_224242_A013299_B013298-A-037420-037721
# 20041028_124231_A013309_B013310-B-038118-038644


cp test.ipa.txt test.ipa
cat test.syl.txt | \
perl -e '
while($l=<>){
  chomp($l);
  @l=split(" ",$l);
  shift @l;
  $ll=join(" ",@l);
  print "$ll\n";
}
' > test.syl
#


OUTPUT_DIR=../ipa2syl

for f in *.ipa; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l zh -threads 8 < $f > ${f%.*}.tok.ipa
done

for f in *.syl; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.syl
done

for f in *.syl; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase ipa syl "${fbase}.clean" 1 80
done

for lang in ipa syl; do
  for f in *.tok.${lang} *.tok.clean.${lang}; do
    outfile="${f%.*}.bpe.${merge_ops}.${lang}"
    ${OUTPUT_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
    echo ${outfile}
  done
done

CUDA_VISIBLE_DEVICES=1 python -m nmt.nmt \
    --out_dir=nmt_model_standard_hparams_ipa2syl \
    --inference_input_file=test_dir/test.tok.clean.bpe.4000.ipa \
    --inference_output_file=output_infer_ipa2syl
#

