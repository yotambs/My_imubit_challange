echo "Imubit challange"
python Imubit_challange.py --opt 'adam' --nn 'wrn' --bsz 16  --loss 'categorical_crossentropy' --epoch 100 > train_output_basline.txt
python Imubit_challange.py --nn 'vgg16' --bsz 16  --loss 'my_loss' --epoch 50 > train_output_13.txt
python Imubit_challange.py --nn 'wrn'   --bsz 16  --loss 'my_loss' --epoch 50 > train_output_14.txt



