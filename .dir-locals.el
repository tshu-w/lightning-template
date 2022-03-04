;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((magit-mode . ((eval . (setq-local magit-git-environment
                                    (append magit-git-environment
                                            '("CONDA_DEFAULT_ENV=template"))))))
 (python-mode . ((lsp-pyright-venv-path . "~/.local/share/conda/envs")
                 (lsp-pyright-stub-path . "~/.local/lib/python-type-stubs"))))
