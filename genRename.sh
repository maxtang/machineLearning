ls *%*  | perl -ne 'chomp; print qq{mv "$_"}; s/%3A/:/g; s/%20/ /g; s/%28/(/g; s/%29/)/g; s/%3F/?/g; s/%2C/,/g; print qq{ "$_"\n};' > rename.sh
