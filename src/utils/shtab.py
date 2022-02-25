# HACK for shtab
# 1. https://github.com/iterative/shtab/pull/66
# 2. https://github.com/omni-us/jsonargparse/issues/127

import shtab
from jsonargparse import ActionConfigFile
from shtab import (
    CHOICE_FUNCTIONS,
    FLAG_OPTION,
    OPTION_END,
    OPTION_MULTI,
    SUPPRESS,
    Choice,
    Template,
    complete2pattern,
    escape_zsh,
    get_public_subcommands,
    log,
    mark_completer,
    wordify,
)

OPTION_MULTI = (*OPTION_MULTI, ActionConfigFile)


@mark_completer("zsh")
def complete_zsh(parser, root_prefix=None, preamble="", choice_functions=None):
    """
    Returns zsh syntax autocompletion script.
    See `complete` for arguments.
    """
    root_prefix = wordify("_shtab_" + (root_prefix or parser.prog))
    root_arguments = []
    subcommands = {}  # {cmd: {"help": help, "arguments": [arguments]}}

    choice_type2fn = {k: v["zsh"] for k, v in CHOICE_FUNCTIONS.items()}
    if choice_functions:
        choice_type2fn.update(choice_functions)

    def format_optional(opt):
        return (
            (
                '{nargs}{options}"[{help}]"'
                if isinstance(opt, FLAG_OPTION)
                else '{nargs}{options}"[{help}]:{dest}:{pattern}"'
            )
            .format(
                nargs=(
                    '"(- :)"'
                    if isinstance(opt, OPTION_END)
                    else '"*"'
                    if isinstance(opt, OPTION_MULTI)
                    else ""
                ),
                options=(
                    "{{{}}}".format(",".join(opt.option_strings))
                    if len(opt.option_strings) > 1
                    else '"{}"'.format("".join(opt.option_strings))
                ),
                help=escape_zsh(opt.help or ""),
                dest=opt.dest,
                pattern=complete2pattern(opt.complete, "zsh", choice_type2fn)
                if hasattr(opt, "complete")
                else (
                    choice_type2fn[opt.choices[0].type]
                    if isinstance(opt.choices[0], Choice)
                    else "({})".format(" ".join(map(str, opt.choices)))
                )
                if opt.choices
                else "_default",
            )
            .replace('""', "")
        )

    def format_positional(opt):
        return '"{nargs}:{help}:{pattern}"'.format(
            nargs={"+": "(*)", "*": "(*):"}.get(opt.nargs, ""),
            help=escape_zsh((opt.help or opt.dest).strip().split("\n")[0]),
            pattern=complete2pattern(opt.complete, "zsh", choice_type2fn)
            if hasattr(opt, "complete")
            else (
                choice_type2fn[opt.choices[0].type]
                if isinstance(opt.choices[0], Choice)
                else "({})".format(" ".join(map(str, opt.choices)))
            )
            if opt.choices
            else "_default",
        )

    for sub in parser._get_positional_actions():
        if not sub.choices or not isinstance(sub.choices, dict):
            # positional argument
            opt = sub
            if opt.help != SUPPRESS:
                root_arguments.append(format_positional(opt))
        else:  # subparser
            log.debug(f"choices:{root_prefix}:{sorted(sub.choices)}")
            public_cmds = get_public_subcommands(sub)
            for cmd, subparser in sub.choices.items():
                if cmd not in public_cmds:
                    log.debug("skip:subcommand:%s", cmd)
                    continue
                log.debug("subcommand:%s", cmd)

                # optionals
                arguments = [
                    format_optional(opt)
                    for opt in subparser._get_optional_actions()
                    if opt.help != SUPPRESS
                ]

                # subcommand positionals
                subsubs = sum(
                    (
                        list(opt.choices)
                        for opt in subparser._get_positional_actions()
                        if isinstance(opt.choices, dict)
                    ),
                    [],
                )
                if subsubs:
                    arguments.append('"1:Sub command:({})"'.format(" ".join(subsubs)))

                # positionals
                arguments.extend(
                    format_positional(opt)
                    for opt in subparser._get_positional_actions()
                    if not isinstance(opt.choices, dict)
                    if opt.help != SUPPRESS
                )

                subcommands[cmd] = {
                    "help": (subparser.description or "").strip().split("\n")[0],
                    "arguments": arguments,
                }
                log.debug("subcommands:%s:%s", cmd, subcommands[cmd])

    log.debug("subcommands:%s:%s", root_prefix, sorted(subcommands))

    # References:
    #   - https://github.com/zsh-users/zsh-completions
    #   - http://zsh.sourceforge.net/Doc/Release/Completion-System.html
    #   - https://mads-hartmann.com/2017/08/06/
    #     writing-zsh-completion-scripts.html
    #   - http://www.linux-mag.com/id/1106/
    return Template(
        """\
#compdef ${prog}
# AUTOMATCALLY GENERATED by `shtab`
${root_prefix}_options_=(
  ${root_options}
)
${root_prefix}_commands_() {
  local _commands=(
    ${commands}
  )
  _describe '${prog} commands' _commands
}
${subcommands}
${preamble}
typeset -A opt_args
local context state line curcontext="$curcontext"
_arguments \\
  $$${root_prefix}_options_ \\
  ${root_arguments} \\
  '(-): :${root_prefix}_commands_' \\
  '(-)*::args:->args'
case $words[1] in
  ${commands_case}
esac"""
    ).safe_substitute(
        root_prefix=root_prefix,
        prog=parser.prog,
        commands="\n    ".join(
            '"{}:{}"'.format(cmd, escape_zsh(subcommands[cmd]["help"]))
            for cmd in sorted(subcommands)
        ),
        root_arguments=" \\\n  ".join(root_arguments),
        root_options="\n  ".join(
            format_optional(opt)
            for opt in parser._get_optional_actions()
            if opt.help != SUPPRESS
        ),
        commands_case="\n  ".join(
            "{cmd_orig}) _arguments ${root_prefix}_{cmd} ;;".format(
                cmd_orig=cmd, cmd=wordify(cmd), root_prefix=root_prefix
            )
            for cmd in sorted(subcommands)
        ),
        subcommands="\n".join(
            """
{root_prefix}_{cmd}=(
  {arguments}
)""".format(
                root_prefix=root_prefix,
                cmd=wordify(cmd),
                arguments="\n  ".join(subcommands[cmd]["arguments"]),
            )
            for cmd in sorted(subcommands)
        ),
        preamble=(
            "\n# Custom Preamble\n" + preamble + "\n# End Custom Preamble\n"
            if preamble
            else ""
        ),
    )


shtab.complete_zsh = complete_zsh
