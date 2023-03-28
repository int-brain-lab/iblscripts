function [ml, ap] = getCraniotomyCoordinates(subject, varargin)
%getCraniotomyCoordinates Get a subject's MLAP craniotomy coordinates.
%   Query Alyx surgeries endpoint for the subject's craniotomy coordinates.
%
%   Inputs (Positional):
%     subject (char) - The subject name.
%
%   Inputs (Optional Name-Value Parameters):
%     procedure (char) - The surgical procedure to filter by. Default is
%       'craniotomy'.
%     name (char) - The craniotomy name.  Default is 'craniotomy_00'.
%     alyx (Alyx) - An instance of Alyx for communicating with the database.
%
%   Outputs:
%     ml (double) - The distance from the midline in mm of the craniotomy
%       centre.
%     ap (double) - The anterio-posterior distance from the lambda of the
%       craniotomy centre, in mm.
%
%   Example 1: Fetch craniotomy coordinates for subject 'SP035'
%     [ml, ap] = getCraniotomyCoordinates('SP035')
%
%   Example 1: Fetch specific craniotomy coordinates
%     [ml, ap] = getCraniotomyCoordinates('SP035', 'name', 'craniotomy_01')

% User parameters
p = inputParser;
p.addParameter('procedure', 'craniotomy', @ischar)
p.addParameter('name', 'craniotomy_00', @ischar)
p.addParameter('alyx', Alyx('',''), @(v)isa(v,'Alyx'))
p.parse(varargin{:})

alyx = p.Results.alyx;
procedure = p.Results.procedure;
if ~alyx.IsLoggedIn
  alyx = alyx.login();
end
surgeries = alyx.getData('surgeries', 'subject', subject, 'procedure', procedure);
assert(~isempty(surgeries), 'no %s surgeries found for subject %s', procedure, subject)
json = surgeries(1).json;
name = p.Results.name;
assert(ismember(name, fieldnames(json)), '"%s" not in surgery JSON', name)
ml = json.(name)(1);
ap = json.(name)(2);
