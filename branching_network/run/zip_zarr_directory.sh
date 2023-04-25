# checks a folder for *.zarr files and zips them, creating .zarr.zip
# they can be read using zarr.ZipStore, without changing code,
# but are consecutive on disk.

directory=$1
echo "ziping all *.zarr folders in directory: $directory"

# ask for confirm to delete
read -p "This will delete the original folders. Are you sure? (y/n) " -n 1 -r
echo ""

# to get relative path in zip, we need to cd into the directory
cd $directory

for f in ./*.zarr; do

    printf "zipping $f ... "


    # equivalent of `7z a -tzip archive.zarr.zip archive.zarr/.`
    # which is suggested in the zarr docs
    fname=$(basename $f)
    cd $f && zip -r -7 "$OLDPWD/$fname.zip" . > /dev/null && cd $OLDPWD

    # print success
    if [ $? -eq 0 ]; then
        echo "done"
        rm -r $f
    else
        echo "failed"
    fi
done
