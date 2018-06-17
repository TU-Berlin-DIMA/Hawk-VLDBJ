#script adopted from http://stackoverflow.com/questions/9387751/calculate-mean-variance-and-range-using-bash-script
BEGIN{
  min = 10000000000000000;
  max = -10000000000000000;       # Initialize to the first value
  sum = 0;
  sum2 = 0;              # Running sum of squares
}
{
    #store values in array, so we can later compute the variance
    arr[NR]=$0;
  #for (n=3; n <= N; n++) {   # Process each value on the line
    if ($0 < min) min = $0;    # Current minimum
    if ($0 > max) max = $0;    # Current maximum
    sum += $0;                # Running sum of values
    #sum2 += $0 * $0;           # Running sum of squares
  #}
  #print $1 ": min=" min ", avg=" sum/(NF-1) ", max=" max ", var=" ((sum*sum) - sum2)/(NF-1);
  #print $0 ": min=" min ", avg=" sum/(NF-1) ", max=" max ", var=" ((sum*sum) - sum2)/(NF-1);
}
END{
#    print "min=" min ", avg=" sum/(NR) ", max=" max ", var=" ((sum*sum) - sum2)/(NR);
    average=sum/(NR)
    for (i=1;i<=NR;i++) {
       variance_sum+=(arr[i]-average)^2
    }
    variance=variance_sum/(NR-1)
	 standard_deviation=sqrt(variance)
	 min_error_bar=average-standard_deviation;
	 max_error_bar=average+standard_deviation;
    print min"\t"max"\t"average"\t"variance"\t"standard_deviation"\t"min_error_bar"\t"max_error_bar;
}

