function show_solve(results) {
    `
        <div id="solve-result">
            <br>
            <h1>该题共有 ${results.length} 解</h1>
            <br>
        {% for result in results %}
            <br>
            <h1>第 {{ loop.index }} 解</h1>
            <br>
            {% for i in range(9) %}
                <div class="l l{{ i }}">
                    {% for j in range(9) %}
                        <span><input readonly type="text" value="{{ result[i][j] or '' }}"></span>
                    {% endfor %}
                </div>
            {% endfor %}
        {% endfor %}
        </div>
    `
};

function solve(is_all) {
    $('#solve-result').remove();
    let rows = [];
    for (let i = 0; i < 9; i++) {
        let row = [];
        $(`.row${i}`).find('input').each(function () {
            let val = $(this).val() ? parseInt($(this).val()) : null;
            row.push(val)
        })
        rows.push(row)
    }
    let data = {
        rows: JSON.stringify(rows),
        is_all
    };
    $.post('/solve', data, function (res) {
        $('#box').append(res)
        // show_solve()
    }, 'json')
};
$(document).ready(function () {
    $('input').focus(function () {
        $(this).attr('readonly', false)
    });

    $('#solve').click(function () {
        solve(true)
    });

    $('#solve-one').click(function () {
        solve(false)
    });

    $('#demo').click(function () {
        let demo = JSON.parse($("input[name=demo]").val());
        for (let i = 0; i < 9; i++) {
            demo[i].forEach(function (item, index) {
                $($(`.row${i}`).find('input').get(index)).val(item);
            })
        }

    })


});
