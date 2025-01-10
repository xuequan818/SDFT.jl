using PkgTemplates
t = Template(;
    user="xuequan818",
    dir="/Users/peacemox/Desktop",
    plugins=[
        License(; name="MIT"),
        Git(; manifest=false, ssh=true),
        GitHubActions(; x86=true, coverage=true),
        Codecov(),
        Documenter{GitHubActions}()
    ])