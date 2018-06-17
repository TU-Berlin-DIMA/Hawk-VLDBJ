$(function() {

    var myCodeMirror = CodeMirror.fromTextArea($('#textareaQuery')[0], {
        lineNumbers: true,
        mode:  "text/x-mysql",
        lineWrapping: true,
    });

    $('#queryResultTable').bootstrapTable('hideLoading');

    if(lastQuery != "") {
        myCodeMirror.getDoc().setValue(lastQuery.replace(new RegExp(':NEWLINE:','g'),"\n").replace(new RegExp(':APOSTROPHE:','g'),"\'"));
    }

    $('#selectPredefinedQuery').on('click', function () {
        var selectedQuery = $('#selectPredefinedQuery option:selected').val();
        myCodeMirror.getDoc().setValue(available_queries[selectedQuery].replace(new RegExp(':NEWLINE:','g'),"\n"));
    });

    $('#submitQuery').on('click', function () {
        var query = myCodeMirror.getDoc().getValue().replace(new RegExp('\n','g'),":NEWLINE:").replace(new RegExp('\'','g'),":APOSTROPHE:");
        var query_name = $("#selectPredefinedQuery option:selected").text();

        $.ajax({
            url: 'exec_query.php',
            data: { 
                "query": query,
                "query_name": query_name,                
                },
            success: function(message) {
                location.reload();
            }
        });       
    });
});