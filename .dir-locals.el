;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((magit-mode . ((eval . (setq-local magit-git-environment
                                    (append magit-git-environment
                                            '("CONDA_DEFAULT_ENV=template"))))))
 (python-mode . ((eval . (setq eglot-workspace-configuration
                               `((:python
                                  :venvPath ,(expand-absolute-name "~/.local/share/conda/envs")
                                  :analysis (:diagnosticMode
                                             "openFilesOnly"
                                             :stubPath
                                             ,(expand-absolute-name "~/.local/lib/python-type-stubs")))))))))
