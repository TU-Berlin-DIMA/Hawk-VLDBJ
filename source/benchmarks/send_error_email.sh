
if [ $# -lt 1 ]; then
        echo 'Missing parameter!'
        echo "Usage: $0 <Your Custom Error Message>"
        exit -1
elif [ $# -gt 1 ]; then
        echo 'Too many parameters!'
        echo "Usage: $0 <Your Custom Error Message>"
        exit -1
fi

echo "Dear Experimentalist,

One of your experiments failed on machine '$HOSTNAME':
$1

Kind regards,
Your experiment script system" | mail -s "Experiment Failed On Machine $HOSTNAME" tobias.behrens@dfki.de
