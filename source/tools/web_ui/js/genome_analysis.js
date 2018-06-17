$(function() {
    $('#viewGenome').on('click', function () {
        var contig_id = $("#contig_id_input").val();
        var start = $("#start_input").val();
        var end = $("#end_input").val();

        $.ajax({
            url: 'create_genome_view_data.php',
            data: { 
                "contigid": contig_id,
                "start": start,                
                "end": end,
                },
            success: function(message) {
                location.reload();
            }
        });       
    });

    var genomeViewMirror = CodeMirror.fromTextArea($('#genomeView')[0], {
        lineNumbers: true,
        //mode:  "text/x-mysql",
        //lineWrapping: true,
        readOnly: true,
    });

    genomeViewMirror.getDoc().setValue(genome_view_data.replace(new RegExp(':NEWLINE:','g'),"\n"));
});