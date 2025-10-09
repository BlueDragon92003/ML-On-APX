# Get the next model number to save this model to.
today= date +"%G%V%w" 
file_name=./models/classification-model-$today-000.pth
number=0

while [ -e "$file_name" ]
do
    printf -v file_name "./models/classification-model-$today-%03d.pth" "$(( ++number ))"
done

# Move the current model to the proper location
current_model="$(realpath "./models/current-classification.pth")"
mv "$current_model" "$file_name"

# Delete all unneeded checkpoints
rm ./models/checkpoint-*-classification.pth
rm ./models/current-classification.pth
