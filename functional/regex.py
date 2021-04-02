from regularize import pattern
from regularize.replace import substitution
from regularize.expression import Pattern

# helper functions from
# https://github.com/georgepsarakis/regularize


class HTMLTag(Pattern):
    def __call__(self, opening=True):
        if opening:
            new = self.literal('<')
        else:
            new = self.literal('</')
        return new.any_of(Pattern.ANY_ASCII_CHARACTER).at_least_one().literal('>')    


def create_html_regex():
    pass


def create_url_regex():
    ascii_alphanumeric = pattern().lowercase_ascii_letters().uppercase_ascii_letters().any_number()
    domain_pattern = ascii_alphanumeric.close_bracket() + ascii_alphanumeric.literal('-').quantify(1, 61)
    domain_pattern += ascii_alphanumeric.close_bracket()
    tld_pattern = pattern().lowercase_ascii_letters(closed=False).uppercase_ascii_letters().quantify(minimum=2)
    subdomain_pattern = domain_pattern.group(name='subdomain', optional=True).literal('.').group(optional=True)
    domain_pattern = subdomain_pattern + domain_pattern.literal('.') + tld_pattern
    scheme_pattern = pattern().literal('http').any_of('s').quantify(minimum=0, maximum=1).group('scheme').literal('://')
    path_pattern = pattern().literal('/').any_number().lowercase_ascii_letters().literal('%-_').quantify(minimum=1).match_all()
    return (scheme_pattern + domain_pattern.group('domain') + path_pattern.group(name='path', optional=True)).case_insensitive()